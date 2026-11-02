// Workgroup-shared regular-tier expert GEMM. See the comment above claude_stage.
// Regular-tier expert GEMM with workgroup-shared operands (claude_regular).
// Same ABI, header words, guards, output layout and partial-tile publication as
// expert_quad_regular_cohort; only the product changes: 16 subgroups share
// double-buffered SLM tiles of 128 weight rows x 64 tokens (each subgroup still
// owns 32 rows x 16 members, four accumulators), instead of every 32-lane
// workgroup loading its own fragments straight from global memory.
#[cube]
#[allow(clippy::too_many_arguments)]
fn claude_stage<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    a: &mut SharedMemory<Vector<F, Const<4>>>,
    b: &mut SharedMemory<Vector<F, Const<4>>>,
    cols4: usize,
    wf4: usize,
    row_base: usize,
    member_base: usize,
    tok_base: usize,
    count: usize,
    k4: usize,
    buf: usize,
) {
    let lane = usize::cast_from(UNIT_POS);
    #[unroll]
    for j in 0..4usize {
        let idx = lane + j * 256;
        let r = idx / 8;
        let c = idx % 8;
        a[buf * 1024 + r * 8 + c] = weight[wf4 + (row_base + r) * cols4 + k4 + c];
    }
    #[unroll]
    for j in 0..2usize {
        let idx = lane + j * 256;
        let t = idx / 8;
        let c = idx % 8;
        let mut v = Vector::<F, Const<4>>::empty().fill(F::new(0.0));
        if tok_base + t < count {
            v = input[(member_base + tok_base + t) * cols4 + k4 + c];
        }
        b[buf * 512 + t * 8 + c] = v;
    }
}
#[cube]
#[allow(clippy::too_many_arguments)]
fn claude_product<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    rows: usize,
    count: usize,
    weight_first: usize,
    group: usize,
    input_member_base: usize,
    gu_prefix: usize,
    #[comptime] down: bool,
    #[comptime] direct: bool,
) {
    let sg = usize::cast_from(PLANE_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let tok_base = usize::cast_from(CUBE_POS_Y) * 64;
    let row_base = usize::cast_from(CUBE_POS_X) * 128;
    let token = tok_base + (sg / 4) * 16;
    let row = row_base + (sg % 4) * 32;
    let cols4 = columns / 4;
    let wf4 = weight_first / 4;
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(1024usize);
    // SAFETY: every operand fragment is loaded before each product.
    let a0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a4 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a5 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a6 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a7 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    if comptime!(direct) {
        // Mesa Xe2 2D block loads (INTEL_CMAT_BLOCK2D): fragments straight
        // from global memory; K is a compile-time constant (constant stride).
        let kc = comptime!(if down { 640usize } else { 2560usize });
        let kc4 = comptime!(kc / 4);
        let abase = wf4 + row * kc4;
        let bbase = (input_member_base + token) * kc4;
        let wlen = weight.len();
        let ilen = input.len();
        let mut k4 = 0usize;
        while k4 < kc4 {
            let ak = abase + k4;
            let bk = bbase + k4;
            cmma::load(&a0, &weight.to_slice().slice(ak, wlen), comptime!(kc as u32));
            cmma::load(&a1, &weight.to_slice().slice(ak + comptime!(8 * kc4), wlen), comptime!(kc as u32));
            cmma::load(&a2, &weight.to_slice().slice(ak + comptime!(16 * kc4), wlen), comptime!(kc as u32));
            cmma::load(&a3, &weight.to_slice().slice(ak + comptime!(24 * kc4), wlen), comptime!(kc as u32));
            cmma::load(&b0, &input.to_slice().slice(bk, ilen), comptime!(kc as u32));
            cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
            cmma::execute::<F, F, f32, f32>(&a1, &b0, &c1, &c1);
            cmma::execute::<F, F, f32, f32>(&a2, &b0, &c2, &c2);
            cmma::execute::<F, F, f32, f32>(&a3, &b0, &c3, &c3);
            k4 += 4;
        }
    } else {
        claude_stage::<F>(input, weight, &mut a, &mut b, cols4, wf4, row_base, input_member_base, tok_base, count, 0, 0);
        sync_cube();
        let mut k = 0usize;
        let mut cur = 0usize;
        while k < columns {
            let nxt = 1usize - cur;
            if k + 32 < columns {
                claude_stage::<F>(input, weight, &mut a, &mut b, cols4, wf4, row_base, input_member_base, tok_base, count, (k + 32) / 4, nxt);
            }
            let ab = cur * 1024 + (sg % 4) * 256;
            let bb = cur * 512 + (sg / 4) * 128;
            cmma::load(&a0, &a.to_slice().slice(ab, a.len()), 32);
            cmma::load(&a1, &a.to_slice().slice(ab + 64, a.len()), 32);
            cmma::load(&a2, &a.to_slice().slice(ab + 128, a.len()), 32);
            cmma::load(&a3, &a.to_slice().slice(ab + 192, a.len()), 32);
            cmma::load(&b0, &b.to_slice().slice(bb, b.len()), 32);
            cmma::load(&a4, &a.to_slice().slice(ab + 4, a.len()), 32);
            cmma::load(&a5, &a.to_slice().slice(ab + 68, a.len()), 32);
            cmma::load(&a6, &a.to_slice().slice(ab + 132, a.len()), 32);
            cmma::load(&a7, &a.to_slice().slice(ab + 196, a.len()), 32);
            cmma::load(&b1, &b.to_slice().slice(bb + 4, b.len()), 32);
            cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
            cmma::execute::<F, F, f32, f32>(&a1, &b0, &c1, &c1);
            cmma::execute::<F, F, f32, f32>(&a2, &b0, &c2, &c2);
            cmma::execute::<F, F, f32, f32>(&a3, &b0, &c3, &c3);
            cmma::execute::<F, F, f32, f32>(&a4, &b1, &c0, &c0);
            cmma::execute::<F, F, f32, f32>(&a5, &b1, &c1, &c1);
            cmma::execute::<F, F, f32, f32>(&a6, &b1, &c2, &c2);
            cmma::execute::<F, F, f32, f32>(&a7, &b1, &c3, &c3);
            sync_cube();
            cur = nxt;
            k += 32;
        }
    }
    // Publication is per subgroup; subgroups wholly past `count` publish nothing.
    let mut scratch = SharedMemory::<f32>::new(2048usize);
    if token < count {
        let prefix = usize::cast_from(map[1028 + group]);
        let max_route = usize::cast_from(map[0]) * 10;
        let mut good = false;
        if token + 16 <= count {
            good = usize::cast_from(map[1540 + prefix + token + lane]) < max_route;
        }
        let all_good = plane_all(good);
        if all_good {
            let mut first = (gu_prefix + token) * rows + row;
            if comptime!(down) {
                first = (prefix + token) * rows + row;
            }
            cmma::store(&mut output.slice_mut(first, output.len()), &c0, u32::cast_from(rows), cmma::MatrixLayout::ColMajor);
            cmma::store(&mut output.slice_mut(first + 8, output.len()), &c1, u32::cast_from(rows), cmma::MatrixLayout::ColMajor);
            cmma::store(&mut output.slice_mut(first + 16, output.len()), &c2, u32::cast_from(rows), cmma::MatrixLayout::ColMajor);
            cmma::store(&mut output.slice_mut(first + 24, output.len()), &c3, u32::cast_from(rows), cmma::MatrixLayout::ColMajor);
            if comptime!(down) && row == 0 {
                let route = usize::cast_from(map[1540 + prefix + token + lane]);
                status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = 2;
            }
        } else {
            crate::strata_matrix::global_small::publish(&c0, &mut scratch, output, map, status, fault, token, row, count, gu_prefix, group, down);
            crate::strata_matrix::global_small::publish(&c1, &mut scratch, output, map, status, fault, token, row + 8, count, gu_prefix, group, down);
            crate::strata_matrix::global_small::publish(&c2, &mut scratch, output, map, status, fault, token, row + 16, count, gu_prefix, group, down);
            crate::strata_matrix::global_small::publish(&c3, &mut scratch, output, map, status, fault, token, row + 24, count, gu_prefix, group, down);
        }
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn claude_regular_cohort(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
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
    #[comptime] direct: bool,
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
    let dim_x = comptime!(if direct { 16u32 } else { 256u32 });
    let dim_y = comptime!(if direct { 16u32 } else { 1u32 });
    if CUBE_DIM_X != dim_x || CUBE_DIM_Y != dim_y || CUBE_DIM_Z != 1u32 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    // Distinct uniform guards precede every untrusted map index and padded
    // arithmetic; never rely on eager boolean lowering to suppress a read.
    if h9 < 1
        || h9 > 8
        || slot >= usize::cast_from(h9)
        || h4 > 512 - h9
        || h11 != 16u32
        || h0 < 1
        || h0 > comptime!(crate::strata_prompt::groups::CAP_U32)
        || h1 != comptime!(if down { 640u32 } else { 2560u32 })
        || h2 != comptime!(if down { 2560u32 } else { 1280u32 })
        || h5 != comptime!(if down { 2u32 } else { 0u32 })
        || group >= 512
        || group != usize::cast_from(h4) + slot
        || members < 32
        || members >= 256
        || members > usize::cast_from(h0)
        || prefix > comptime!(crate::strata_prompt::groups::CAPACITY)
        || prefix % 16 != 0
        || raw > 8
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    let padded = (members + 15) / 16 * 16;
    if padded > comptime!(crate::strata_prompt::groups::CAPACITY) - prefix || map.len() != 65536 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if map[0] != h0
        || map[1] != 0
        || map[2] != h0
        || map[3] < 1
        || map[3] > 512
        || group >= usize::cast_from(map[3])
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if map[4 + group] >= 512
        || map[516 + group] != count
        || usize::cast_from(map[1028 + group]) > usize::cast_from(h0) * 10
        || members > usize::cast_from(h0) * 10 - usize::cast_from(map[1028 + group])
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if usize::cast_from(CUBE_POS_Y) * 64 >= members {
        terminate!();
    }
    let mut input_prefix=prefix;
    if h13>comptime!(if cfg!(feature="strata-layer-gather") {1u32}else{0u32}) || h14!=0 || h15!=0 {
        if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
    }
    // Only GU reads global X. Keep each terminating guard at the outer
    // kernel scope: nested early exits otherwise form invalid SPIR-V CFG.
    if comptime!(!down) {
        let global=usize::cast_from(map[comptime!(crate::strata_prompt::groups::LAYER_PREFIX)+group]);
        let total=usize::cast_from(map[comptime!(crate::strata_prompt::groups::LAYER_PREFIX+512)]);
        let extent=(members+31)/32*32;
        if h13==1 && (total>comptime!(crate::strata_prompt::groups::PADDED_ROWS) || global%32!=0 || global>total) {
            if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
        }
        if h13==1 && extent>total-global {
            if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
        }
        let mut next=total;
        if group+1<usize::cast_from(map[3]) {next=usize::cast_from(map[comptime!(crate::strata_prompt::groups::LAYER_PREFIX)+group+1]);}
        if h13==1 && ((group==0 && global!=0) || global+extent!=next || input.len()!=comptime!(crate::strata_prompt::groups::PADDED_ROWS*640)) {
            if UNIT_POS==0 {fault[0].fetch_or(128);}terminate!();
        }
        if h13==1 {input_prefix=global;}
    }
    let weight_first = slot * 4915200 + comptime!(if down { 3276800usize } else { 0usize });
    claude_product::<f16>(
        input, weight, output, map, status, fault, columns, rows, members, weight_first,
        group, input_prefix, prefix, down, direct,
    );
}

pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    words: &[u32; 48],
    down: bool,
    _tile: usize,
) -> Result<()> {
    crate::strata_matrix::native::compatible(client, false)?;
    let padded = crate::strata_prompt::experts::regular::validate_words(words)?;
    let [input, weight, output, map, status, fault] = h;
    if words[11] != 16
        || words[1] != if down { 640 } else { 2560 }
        || words[2] != if down { 2560 } else { 1280 }
        || words[5] != if down { 2 } else { 0 }
        || words[3] != 0
        || input.size_in_used() < (padded * words[1] as usize * 2) as u64
        || weight.size_in_used()
            < u64::from(words[9]) * crate::strata_prompt::experts::cohorts::IMAGE_BYTES
        || output.size_in_used()
            != if down {
                crate::strata_prompt::groups::PART_BYTES
            } else {
                crate::strata_prompt::groups::GU_BYTES
            }
        || map.size_in_used() != (crate::strata_prompt::groups::MAP_WORDS * 4) as u64
        || status.size_in_used() != (crate::strata_prompt::groups::STATUS_WORDS * 4) as u64
        || fault.size_in_used() != 4
        || [&input, &weight, &output, &map, &status, &fault]
            .iter()
            .any(|v| v.stream != input.stream)
    {
        return Err(Error::InvalidBufferLength);
    }

    for v in [&input, &weight] {
        if v.size_in_used() % 8 != 0 || v.offset_start.unwrap_or(0) % 8 != 0 {
            return Err(Error::InvalidBufferLength);
        }
    }
    let ia = unsafe { ArrayArg::from_raw_parts(input.clone(), input.size_in_used() as usize / 2) };
    let wa =
        unsafe { ArrayArg::from_raw_parts(weight.clone(), weight.size_in_used() as usize / 2) };
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let ma = crate::resident_util::raw_arg(map.clone(), map.size_in_used() as usize / 4)?;
    let sa = crate::resident_util::raw_arg(status.clone(), status.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    let hw = &client.properties().hardware;
    if hw.max_units_per_cube < 256 || hw.max_cube_dim.0 < 256 || hw.max_shared_memory_size < 1024 {
        return Err(Error::LimitExceeded("balanced cohort native geometry"));
    }
    // Mesa with Xe2 2D block loads: direct fragments, 16 x 16 opt-in workgroup.
    let direct = std::env::var_os("INTEL_CMAT_BLOCK2D").is_some()
        && std::env::var_os("KUNO_CLAUDE_EXPERT_DIRECT_OFF").is_none();
    let settings = matrix_settings().cube_dim(if direct { CubeDim::new_2d(16, 16) } else { CubeDim::new_1d(256) });
    let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
    launcher.with_scope(|scope| scope.device_properties(client.properties()));
    let [ia, wa, oa, ma, sa, fa] = register_arrays(&mut launcher, [ia, wa, oa, ma, sa, fa]);
    register_words(&mut launcher, words);
    // The original generated kernel's runtime-u32 compilation argument is ().
    // Construct that SAME kernel with the same authentic array compilation
    // arguments and settings; only scalar serialization is performed in bulk.
    let kernel = claude_regular_cohort::ClaudeRegularCohort::<WgpuRuntime>::new(
        settings,
        client.clone(),
        ia,
        wa,
        oa,
        ma,
        sa,
        fa,
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        down,
        direct,
    );
    unsafe {
        launcher.launch_unchecked(
            CubeCount::Static(
                (if down { 2560usize } else { 1280usize } / 128) as u32,
                words[10].div_ceil(64),
                words[9],
            ),
            kernel,
            client,
        );
    }
    crate::strata_prefill_profile::count(
        if down {
            "claude_regular_cohort_down_launches"
        } else {
            "claude_regular_cohort_gu_launches"
        },
        1,
    );
    crate::strata_prefill_profile::count(
        "claude_regular_cohort_matrix_workgroups",
        (if down { 2560u64 } else { 1280u64 }) / 128
            * u64::from(words[10].div_ceil(64))
            * u64::from(words[9]),
    );
    crate::strata_prefill_profile::count(
        "claude_regular_cohort_active_tile_workgroups",
        (if down { 2560u64 } else { 1280u64 }) / 128
            * (0..words[9] as usize)
                .map(|i| u64::from(words[17 + i * 4].div_ceil(64)))
                .sum::<u64>(),
    );
    Ok(())
}
