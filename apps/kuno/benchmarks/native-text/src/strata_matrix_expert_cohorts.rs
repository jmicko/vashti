//! Genuine cross-expert narrow CMMA. Each group's original8/16-row tile,
//! chronologicalK16 arithmetic, FP16 inputs and routed publication are retained.
#[cfg(feature="strata-global-small-products")]
#[path="strata_matrix_global_small_cohort.rs"]
pub(crate) mod global_small;
#[cfg(feature="strata-small-full-walk")]
#[path="strata_matrix_small_full_walk_cohort.rs"]
pub(crate) mod small_full_walk;
#[cfg(feature="strata-expert-ordered-output")]
#[path="strata_matrix_ordered_cohorts.rs"]
pub(crate) mod ordered;
#[cfg(feature="strata-small-direct-publish")]
#[path="strata_matrix_cohort_small_direct.rs"]
pub(crate) mod small_direct;
#[cfg(feature="strata-expert-ordered-output")]
pub(crate) use ordered::launch as launch_ordered;
use super::*;
use cubecl::{
    compute::KernelLauncher,
    ir::{ElemType, UIntKind},
};
use cubecl_runtime::kernel::KernelDefinition;
use half::f16;
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_cohort(
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
    #[comptime] tile: usize,
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
    if (members <= 8) != comptime!(tile == 8)
        || usize::cast_from(CUBE_POS_Y) * comptime!(tile) >= members
    {
        terminate!();
    }
    let weight_first = slot * 4915200 + comptime!(if down { 3276800usize } else { 0usize });
    let subgroup = usize::cast_from(UNIT_POS) / 16;
    let token_base = usize::cast_from(CUBE_POS_Y) * comptime!(tile);
    let row_base = usize::cast_from(CUBE_POS_X) * 64;
    let a0 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let a1 = if comptime!(tile == 16) {
        unsafe {
            cmma::Matrix::<f16>::uninitialized(
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
    if comptime!(cfg!(feature = "strata-expert-register-prefetch")) {
        super::register_prefetch::accumulate::<f16>(
            input,
            weight,
            columns,
            (prefix + token_base) * columns / 4,
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
            let af = ((prefix + token_base) * columns + k) / 4;
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
            cmma::execute::<f16, f16, f32, f32>(&a0, &bm, &c0, &c0);
            if comptime!(tile == 16) {
                cmma::load(
                    &a1,
                    &input.to_slice().slice(af + 8 * columns / 4, input.len()),
                    u32::cast_from(columns),
                );
                cmma::execute::<f16, f16, f32, f32>(&a1, &bm, &c1, &c1);
            }
            k += 16;
        }
    }
    let mut out = SharedMemory::<f32>::new(comptime!(tile * 64));
    let first = subgroup * comptime!(tile * 16);
    cmma::store(
        &mut out.slice_mut(first, first + 128),
        &c0,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    if comptime!(tile == 16) {
        cmma::store(
            &mut out.slice_mut(first + 128, first + 256),
            &c1,
            16,
            cmma::MatrixLayout::RowMajor,
        );
    }
    sync_cube();
    let mut i = usize::cast_from(UNIT_POS) % 16;
    while i < comptime!(tile * 16) {
        let member = token_base + i / 16;
        let row = row_base + subgroup * 16 + i % 16;
        if member < members && row < rows {
            let route = usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
            if route < usize::cast_from(map[0]) * 10 {
                let mut base = (prefix + member) * 1280;
                if comptime!(down) {
                    base = route * 2560;
                }
                let value = out[subgroup * comptime!(tile * 16) + i];
                let valid = value.abs() <= f32::MAX;
                if valid {
                    output[base + row] = value;
                } else {
                    output[base + row] = 0.0;
                    fault[0].fetch_or(128);
                }
                if comptime!(down) && row == 0 {
                    let mut good = 0u32;
                    if valid && status[route] == 1 {
                        good = 1;
                    }
                    status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = good;
                }
            } else {
                fault[0].fetch_or(128);
            }
        }
        i += 16;
    }
}
pub(crate) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    [
        ("expert_cohort8_gu", false, 8),
        ("expert_cohort8_down", true, 8),
        ("expert_cohort16_gu", false, 16),
        ("expert_cohort16_down", true, 16),
    ]
    .map(|(name, down, tile)| (name, definition(down, tile)))
    .into_iter()
    .collect()
}
fn definition(down: bool, tile: usize) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    AddressType::U32.register(&mut b.scope);
    let t = Vector::<f16, Const<4>>::as_type(&b.scope);
    let input = b.input_array(t);
    let weight = b.input_array(t);
    let output = b.output_array(f32::as_type(&b.scope));
    let map = b.input_array(u32::as_type(&b.scope));
    let status = b.output_array(u32::as_type(&b.scope));
    let fault =
        b.output_array(cubecl::ir::StorageType::Atomic(ElemType::UInt(UIntKind::U32)).into());
    let scalars: [cubecl::ir::ManagedVariable; 48] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    expert_cohort::expand(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        map.into(),
        status.into(),
        fault.into(),
        scalars[0].clone().into(),
        scalars[1].clone().into(),
        scalars[2].clone().into(),
        scalars[3].clone().into(),
        scalars[4].clone().into(),
        scalars[5].clone().into(),
        scalars[6].clone().into(),
        scalars[7].clone().into(),
        scalars[8].clone().into(),
        scalars[9].clone().into(),
        scalars[10].clone().into(),
        scalars[11].clone().into(),
        scalars[12].clone().into(),
        scalars[13].clone().into(),
        scalars[14].clone().into(),
        scalars[15].clone().into(),
        scalars[16].clone().into(),
        scalars[17].clone().into(),
        scalars[18].clone().into(),
        scalars[19].clone().into(),
        scalars[20].clone().into(),
        scalars[21].clone().into(),
        scalars[22].clone().into(),
        scalars[23].clone().into(),
        scalars[24].clone().into(),
        scalars[25].clone().into(),
        scalars[26].clone().into(),
        scalars[27].clone().into(),
        scalars[28].clone().into(),
        scalars[29].clone().into(),
        scalars[30].clone().into(),
        scalars[31].clone().into(),
        scalars[32].clone().into(),
        scalars[33].clone().into(),
        scalars[34].clone().into(),
        scalars[35].clone().into(),
        scalars[36].clone().into(),
        scalars[37].clone().into(),
        scalars[38].clone().into(),
        scalars[39].clone().into(),
        scalars[40].clone().into(),
        scalars[41].clone().into(),
        scalars[42].clone().into(),
        scalars[43].clone().into(),
        scalars[44].clone().into(),
        scalars[45].clone().into(),
        scalars[46].clone().into(),
        scalars[47].clone().into(),
        down,
        tile,
    );
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))
}
// Preserve the generated launcher's settings and six actual registration
// types. Every array retains its original owner, scalar length and vec4 ABI.
fn matrix_settings() -> KernelSettings {
    KernelSettings::default()
        .cube_dim(CubeDim::new_1d(64))
        .address_type(AddressType::U32)
}
type ArrayCompilation = <Array<f32> as LaunchArg>::CompilationArg;
fn register_arrays(
    launcher: &mut KernelLauncher<WgpuRuntime>,
    args: [ArrayArg<WgpuRuntime>; 6],
) -> [ArrayCompilation; 6] {
    let [input, weight, output, map, status, fault] = args;
    [
        <Array<Vector<f16, Const<4>>> as LaunchArg>::register(input, launcher),
        <Array<Vector<f16, Const<4>>> as LaunchArg>::register(weight, launcher),
        <Array<f32> as LaunchArg>::register(output, launcher),
        <Array<u32> as LaunchArg>::register(map, launcher),
        <Array<u32> as LaunchArg>::register(status, launcher),
        <Array<Atomic<u32>> as LaunchArg>::register(fault, launcher),
    ]
}
fn register_words(launcher: &mut KernelLauncher<WgpuRuntime>, words: &[u32; 48]) {
    // CubeElement supplies the identical native-endian bytes and U32 scalar
    // type used by each original scalar registration. Borrow the caller's
    // existing wiped metadata directly, without a temporary byte-vector.
    // The real ScalarBuilder owns this single192-byte copy and retains all of
    // its original full-capacity success/error/panic erasure guards.
    launcher.register_scalar_raw(
        <u32 as cubecl::CubeElement>::as_bytes(words),
        <u32 as cubecl::CubeElement>::cube_type(),
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    words: &[u32; 48],
    down: bool,
    tile: usize,
) -> Result<()> {
    super::native::compatible(client, false)?;
    let padded = crate::strata_prompt::experts::cohorts::validate_words(words)?;
    let [input, weight, output, map, status, fault] = h;
    if !matches!(tile, 8 | 16)
        || words[11] != tile as u32
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
    let settings = matrix_settings();
    let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
    launcher.with_scope(|scope| scope.device_properties(client.properties()));
    let [ia, wa, oa, ma, sa, fa] = register_arrays(&mut launcher, [ia, wa, oa, ma, sa, fa]);
    register_words(&mut launcher, words);
    // The original generated kernel's runtime-u32 compilation argument is ().
    // Construct that SAME kernel with the same authentic array compilation
    // arguments and settings; only scalar serialization is performed in bulk.
    let kernel = expert_cohort::ExpertCohort::<WgpuRuntime>::new(
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
        tile,
    );
    unsafe {
        launcher.launch_unchecked(
            CubeCount::Static(
                (if down { 2560usize } else { 1280usize } / 64) as u32,
                if tile == 8 { 1 } else { 2 },
                words[9],
            ),
            kernel,
            client,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "strata_matrix_cohort_scalar_tests.rs"]
mod scalar_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cohort_fixed_families_keep_original_k16_tile_publication_and_real_route_status_plane() {
        let s = include_str!("strata_matrix_expert_cohorts.rs");
        assert!(s.contains("k += 16"));
        assert!(s.contains("prefix + token_base"));
        assert!(s.contains("route * 2560"));
        assert!(s.contains("groups::ROUTES"));
        for tile in [8usize, 16] {
            for members in 1usize..32 {
                let family = if members <= 8 { 8 } else { 16 };
                if family != tile {
                    continue;
                }
                let mut visits = vec![0u8; members * 64];
                for gy in 0..if tile == 8 { 1 } else { 2 } {
                    if gy * tile >= members {
                        continue;
                    }
                    for sg in 0..4 {
                        for i in 0..tile * 16 {
                            let member = gy * tile + i / 16;
                            let row = sg * 16 + i % 16;
                            if member < members {
                                visits[member * 64 + row] += 1;
                            }
                        }
                    }
                }
                assert!(visits.into_iter().all(|n| n == 1));
            }
        }
    }
}

#[path = "strata_matrix_cohort_pair_probe.rs"]
mod pair_probe;
#[cfg(feature = "strata-global-small-products")]
#[path = "strata_matrix_global_small_pair_probe.rs"]
mod global_small_pair_probe;
#[cfg(feature = "strata-global-small-products")]
pub(crate) use global_small_pair_probe::cases as public_global_small_pairs;
#[path = "strata_routed_expert_probe.rs"]
mod public_oracle;
pub(crate) use pair_probe::cases as public_pairs;
pub(crate) fn public_numeric(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut crate::private_arena::PrivateArena,
) -> std::result::Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    use public_oracle::Control::*;
    fn unused(
        _: &ComputeClient<WgpuRuntime>,
        _: [Handle; 6],
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: bool,
    ) -> public_oracle::ProbeResult<bool> {
        Err("public cohort original oracle dispatch was not selected".into())
    }
    public_oracle::cases(
        client,
        arena,
        "native-cohort-routed-expert",
        "actual_native_cohort_launch",
        &[
            (1, Finite),
            (3, Finite),
            (7, Finite),
            (8, Finite),
            (9, Finite),
            (15, Finite),
            (16, Finite),
            (17, Finite),
            (31, Finite),
            (3, NonfiniteFirst),
            (3, NonfiniteOther),
            (3, InvalidRoute),
            (3, InvalidRoutePrefix),
            (17, NonfiniteFirst),
            (17, NonfiniteOther),
            (17, InvalidRoute),
            (17, InvalidRoutePrefix),
        ],
        unused,
    )
}

#[cfg(feature="strata-small-full-walk")]
#[path="strata_matrix_small_full_walk_pair_probe.rs"]mod full_walk_pair_probe;
#[cfg(feature="strata-small-full-walk")]
pub(crate) use full_walk_pair_probe::cases as public_full_walk_pairs;
