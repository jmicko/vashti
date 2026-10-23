#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_quant_cohort(
    input: &Array<Vector<f16, Const<4>>>,
    packed: &Array<u32>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    strata_grid: &Array<u32>,
    h0: u32,
    h1: u32,
    h2: u32,
    h3: u32,
    h4: u32,
    h5: u32,
    h6: u32,
    h7: u32,
    h8: u32,
    h9: u32,
    h10: u32,
    h11: u32,
    h12: u32,
    h13: u32,
    h14: u32,
    h15: u32,
    m0: u32,
    m1: u32,
    m2: u32,
    m3: u32,
    m4: u32,
    m5: u32,
    m6: u32,
    m7: u32,
    m8: u32,
    m9: u32,
    m10: u32,
    m11: u32,
    m12: u32,
    m13: u32,
    m14: u32,
    m15: u32,
    m16: u32,
    m17: u32,
    m18: u32,
    m19: u32,
    m20: u32,
    m21: u32,
    m22: u32,
    m23: u32,
    m24: u32,
    m25: u32,
    m26: u32,
    m27: u32,
    m28: u32,
    m29: u32,
    m30: u32,
    m31: u32,
    #[comptime] down: bool,
    #[comptime] tile: usize,
    #[comptime] dtype: u32,
) {
    let slot = usize::cast_from(CUBE_POS_Z);
    let mut group = 0u32;
    let mut count = 0u32;
    let mut prefix = 0u32;
    let mut raw = 0u32;
    if slot == 0 {
        group = m0;
        count = m1;
        prefix = m2;
        raw = m3;
    }
    if slot == 1 {
        group = m4;
        count = m5;
        prefix = m6;
        raw = m7;
    }
    if slot == 2 {
        group = m8;
        count = m9;
        prefix = m10;
        raw = m11;
    }
    if slot == 3 {
        group = m12;
        count = m13;
        prefix = m14;
        raw = m15;
    }
    if slot == 4 {
        group = m16;
        count = m17;
        prefix = m18;
        raw = m19;
    }
    if slot == 5 {
        group = m20;
        count = m21;
        prefix = m22;
        raw = m23;
    }
    if slot == 6 {
        group = m24;
        count = m25;
        prefix = m26;
        raw = m27;
    }
    if slot == 7 {
        group = m28;
        count = m29;
        prefix = m30;
        raw = m31;
    }

    let columns = comptime!(if down { 640usize } else { 2560usize });
    let rows = comptime!(if down { 2560usize } else { 1280usize });
    let members = usize::cast_from(count);
    let group = usize::cast_from(group);
    let prefix = usize::cast_from(prefix);
    // Distinct uniform guards precede every untrusted map index and padded
    // arithmetic; never rely on eager boolean lowering to suppress a read.
    if h9 < 1
        || h9 > 8
        || slot >= usize::cast_from(h9)
        || h4 > 512 - h9
        || h11 != comptime!(tile as u32)
        || h0 < 1
        || h0 > 4096
        || h1 != comptime!(if down { 640u32 } else { 2560u32 })
        || h2 != comptime!(if down { 2560u32 } else { 1280u32 })
        || h5 != comptime!(if down { 2u32 } else { 0u32 })
        || group >= 512
        || group != usize::cast_from(h4) + slot
        || members < 1
        || members >= 32
        || members > usize::cast_from(h0)
        || prefix > 4096
        || prefix % 8 != 0
        || raw > 8
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    let padded = (members + comptime!(tile) - 1) / comptime!(tile) * comptime!(tile);
    if padded > 4096 - prefix || map.len() != 65536 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    // Match original native header validation: all lanes check the same
    // immutable map cells. A genuine subgroup vote keeps the uniform accepted
    // continuation visible to shared-memory lifetime analysis.
    if !plane_all(
        !(map[0] != h0
            || map[1] != 0
            || map[2] != h0
            || map[3] < 1
            || map[3] > 512
            || group >= usize::cast_from(map[3])),
    ) {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if !plane_all(
        !(map[4 + group] >= 512
            || map[516 + group] != count
            || usize::cast_from(map[1028 + group]) > usize::cast_from(h0) * 10
            || members > usize::cast_from(h0) * 10 - usize::cast_from(map[1028 + group])),
    ) {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if (members <= 8) != comptime!(tile == 8)
        || (members > 8 && members <= 16) != comptime!(tile == 16)
        || (members > 16) != comptime!(tile == 32)
    {
        terminate!();
    }

    let row_bytes = comptime!(crate::strata_matrix::quant_cohort::quant_row_bytes(
        dtype, down
    ));
    if h3 != comptime!(row_bytes as u32)
        || h6 == 0
        || h6 > 1048576
        || h6 != h7
        || h8 <= 2 * h6
        || h8 > 1048576
        || h12 > 1
        || h13 != 0
        || h14 != 0
        || h15 != 0
        || packed.len() != 20971520
        || strata_grid.len() != 8192
        || status.len() != 131072
        || fault.len() != 1
        || output.len() != comptime!(if down { 104857600usize } else { 5242880usize })
        || input.len() < (prefix + padded) * columns / 4
        || usize::cast_from(h8) * 8 > packed.len()
        || CUBE_POS_X >= comptime!((if down { 2560u32 } else { 1280u32 }) / 64)
        || CUBE_POS_Y != 0
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if comptime!(down) {
        if h8 - 2 * h6 != comptime!((2560 * row_bytes / 4) as u32) {
            if UNIT_POS == 0 {
                fault[0].fetch_or(128);
            }
            terminate!();
        }
    } else {
        if h6 != comptime!((640 * row_bytes / 4) as u32) {
            if UNIT_POS == 0 {
                fault[0].fetch_or(128);
            }
            terminate!();
        }
    }
    let sg = usize::cast_from(PLANE_POS);
    let row_base = usize::cast_from(CUBE_POS_X) * 64 + sg * 16;
    let mut source_first = slot * usize::cast_from(h8) * 4;
    if comptime!(down) {
        source_first += usize::cast_from(h6 + h7) * 4;
    }
    // Gather/activation initialize every padded input row. Load A directly,
    // as the accepted global narrow products do; only decoded weights need SLM.
    let mut b = SharedMemory::<Vector<f16, Const<4>>>::new(4usize * 2 * 376);
    let am0 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let am1 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let am2 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let am3 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let bm = unsafe {
        cmma::Matrix::<f16>::uninitialized(
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
    // One K64 weight panel per bank. Direct input loads preserve original
    // row strides and all four chronological K16 products. The final
    // iteration drains the last complete panel without staging another.
    let mut producer_k = 0usize;
    let mut producer_bank = 0usize;
    while producer_k <= columns {
        let previous_bank = 1usize - producer_bank;
        if producer_k < columns {
            crate::strata_matrix::quant_cohort::stage(
                packed,
                strata_grid,
                &mut b,
                fault,
                row_base,
                source_first,
                usize::cast_from(h6) * 4,
                producer_k,
                producer_bank,
                down,
                dtype,
            );
        }
        if producer_k == 0 {
            sync_plane();
        } else {
            if comptime!(tile == 32) {
                for step in 0..4usize {
                    let af = (prefix * columns + producer_k - 64 + step * 16) / 4;
                    let bf = (sg * 2 + previous_bank) * 376 + step * 94;
                    cmma::load(
                        &am0,
                        &input.to_slice().slice(af, input.len()),
                        u32::cast_from(columns),
                    );
                    cmma::load(&bm, &b.to_slice().slice(bf, b.len()), 24);
                    cmma::execute::<f16, f16, f32, f32>(&am0, &bm, &c0, &c0);
                    if comptime!(tile >= 16) {
                        cmma::load(
                            &am1,
                            &input.to_slice().slice(af + 8 * columns / 4, input.len()),
                            u32::cast_from(columns),
                        );
                        cmma::execute::<f16, f16, f32, f32>(&am1, &bm, &c1, &c1);
                    }
                    if comptime!(tile == 32) {
                        cmma::load(
                            &am2,
                            &input.to_slice().slice(af + 16 * columns / 4, input.len()),
                            u32::cast_from(columns),
                        );
                        cmma::load(
                            &am3,
                            &input.to_slice().slice(af + 24 * columns / 4, input.len()),
                            u32::cast_from(columns),
                        );
                        cmma::execute::<f16, f16, f32, f32>(&am2, &bm, &c2, &c2);
                        cmma::execute::<f16, f16, f32, f32>(&am3, &bm, &c3, &c3);
                    }
                }
            } else {
                #[unroll]
                for step in 0..4usize {
                    let af = (prefix * columns + producer_k - 64 + step * 16) / 4;
                    let bf = (sg * 2 + previous_bank) * 376 + step * 94;
                    cmma::load(
                        &am0,
                        &input.to_slice().slice(af, input.len()),
                        u32::cast_from(columns),
                    );
                    cmma::load(&bm, &b.to_slice().slice(bf, b.len()), 24);
                    cmma::execute::<f16, f16, f32, f32>(&am0, &bm, &c0, &c0);
                    if comptime!(tile >= 16) {
                        cmma::load(
                            &am1,
                            &input.to_slice().slice(af + 8 * columns / 4, input.len()),
                            u32::cast_from(columns),
                        );
                        cmma::execute::<f16, f16, f32, f32>(&am1, &bm, &c1, &c1);
                    }
                    if comptime!(tile == 32) {
                        cmma::load(
                            &am2,
                            &input.to_slice().slice(af + 16 * columns / 4, input.len()),
                            u32::cast_from(columns),
                        );
                        cmma::load(
                            &am3,
                            &input.to_slice().slice(af + 24 * columns / 4, input.len()),
                            u32::cast_from(columns),
                        );
                        cmma::execute::<f16, f16, f32, f32>(&am2, &bm, &c2, &c2);
                        cmma::execute::<f16, f16, f32, f32>(&am3, &bm, &c3, &c3);
                    }
                }
            }
            sync_plane();
        }
        producer_bank = previous_bank;
        producer_k += 64;
    }
    // Explicit-layout devices may overlay publication scratch with B. This
    // is a cross-subgroup reuse: join the whole workgroup once before freeing
    // the final B lifetime, not merely each subgroup's last panel handoff.
    sync_cube();
    unsafe {
        b.free();
    }
    let mut scratch = SharedMemory::<f32>::new(4usize * 128);
    crate::strata_matrix::small_direct::publish(
        &c0,
        &mut scratch,
        output,
        map,
        status,
        fault,
        0,
        row_base,
        members,
        prefix,
        group,
        down,
    );
    if comptime!(tile >= 16) {
        crate::strata_matrix::small_direct::publish(
            &c1,
            &mut scratch,
            output,
            map,
            status,
            fault,
            8,
            row_base,
            members,
            prefix,
            group,
            down,
        );
    }
    if comptime!(tile == 32) {
        crate::strata_matrix::small_direct::publish(
            &c2,
            &mut scratch,
            output,
            map,
            status,
            fault,
            16,
            row_base,
            members,
            prefix,
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
            24,
            row_base,
            members,
            prefix,
            group,
            down,
        );
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage(
    packed: &Array<u32>,
    grid: &Array<u32>,
    b: &mut SharedMemory<Vector<f16, Const<4>>>,
    fault: &mut Array<Atomic<u32>>,
    row_base: usize,
    source_first: usize,
    up_offset: usize,
    k: usize,
    bank: usize,
    #[comptime] down: bool,
    #[comptime] dtype: u32,
) {
    let sg = usize::cast_from(PLANE_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    // Each half subgroup reads contiguous K64 of one encoded row. Every
    // eight-value group has exactly one writer; two rows advance per round.
    for round in 0..8usize {
        let local_row = round * 2 + lane / 8;
        let row = row_base + local_row;
        let col = lane % 8 * 8;
        let first = (sg * 2 + bank) * 376 + (col / 16) * 94 + local_row * 6 + (col % 16) / 4;
        let mut packed_row = source_first
            + row
                * comptime!(crate::strata_matrix::quant_cohort::quant_row_bytes(
                    dtype, down
                ));
        if comptime!(!down) {
            packed_row = source_first
                + (row / 2)
                    * comptime!(crate::strata_matrix::quant_cohort::quant_row_bytes(
                        dtype, false
                    ));
            if row % 2 != 0 {
                packed_row += up_offset;
            }
        }
        crate::strata_matrix::quant_cohort::decoder::store8(
            packed,
            grid,
            packed_row,
            k + col,
            b,
            first,
            fault,
            dtype,
        );
        // Pitch24 gives every16-value matrix column a16-byte padding cell.
        // The final column has no trailing padding, so each K16 tile occupies
        // 376 halves and starts at a16-byte aligned address. Initialize every
        // padding vector exactly once, including both lookahead banks.
        if col % 16 == 0 && local_row < 15 {
            b[first + 4] = Vector::<f16, Const<4>>::new(f16::new(0.0));
            b[first + 5] = Vector::<f16, Const<4>>::new(f16::new(0.0));
        }
    }
}
