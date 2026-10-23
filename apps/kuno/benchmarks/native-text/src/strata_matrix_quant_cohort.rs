//! Fused exact quantized small-cohort CMMA over an authenticated compact slab.
//! Source preparation/scheduling owns all model and RAW accesses. This module
//! decodes only the original quant bytes, rounds once to exact original F16,
//! and consumes each output's K16 contributions in chronological order.
use super::*;
use cubecl::{
    compute::KernelLauncher,
    ir::{ElemType, UIntKind},
};
use cubecl_runtime::kernel::KernelDefinition;
use half::f16;
#[path = "strata_matrix_quant_cohort_decoder.rs"]
pub(crate) mod decoder;
include!("strata_matrix_quant_cohort_kernel.rs");
#[path = "strata_matrix_quant_cohort_validation.rs"]
mod validation;
pub(crate) use validation::validate as validate_sources;
#[path = "strata_matrix_quant_cohort_probe.rs"]
mod probe;
pub(crate) use probe::cases as public_numeric;
#[cfg(test)]
#[path = "strata_matrix_quant_cohort_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "strata_matrix_quant_half_tests.rs"]
mod native_half_tests;
#[path = "strata_matrix_quant_half_probe.rs"]
pub(crate) mod half_probe;
#[path = "strata_matrix_quant_half_validation.rs"]
mod half_validation;
pub(crate) use half_validation::validate as validate_half_sources;

pub(crate) const fn supported(dtype: u32, down: bool) -> bool {
    if down {
        matches!(dtype, 20 | 42)
    } else {
        matches!(dtype, 18 | 21 | 22 | 23)
    }
}
pub(crate) const fn quant_row_bytes(dtype: u32, down: bool) -> usize {
    match (dtype, down) {
        (18, false) => 980,
        (21, false) => 1100,
        (22, false) => 820,
        (23, false) => 1360,
        (20, true) => 360,
        (42, true) => 180,
        _ => 0,
    }
}
pub(crate) fn role_shape(words: &[u32; 48], dtype: u32, down: bool) -> bool {
    let g = words[6] as usize;
    let total = words[8] as usize;
    if !supported(dtype, down) || g > 1048576 || words[7] != words[6] || total <= g * 2 {
        return false;
    }
    let d = total - g * 2;
    if ![640 * 980 / 4, 640 * 1100 / 4, 640 * 820 / 4, 640 * 1360 / 4].contains(&g)
        || ![2560 * 360 / 4, 2560 * 180 / 4].contains(&d)
    {
        return false;
    }
    if down {
        d == 2560 * quant_row_bytes(dtype, true) / 4
    } else {
        g == 640 * quant_row_bytes(dtype, false) / 4
    }
}
pub(crate) fn definitions() -> Vec<(String, KernelDefinition)> {
    let mut out = Vec::new();
    for dtype in [18u32, 21, 22, 23, 20, 42] {
        let down = matches!(dtype, 20 | 42);
        for tile in [8usize, 16, 32] {
            out.push((
                format!("quant_cohort_dtype{dtype}_tile{tile}"),
                definition(down, tile, dtype),
            ));
        }
    }
    out
}
fn definition(down: bool, tile: usize, dtype: u32) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    AddressType::U32.register(&mut b.scope);
    let t = Vector::<f16, Const<4>>::as_type(&b.scope);
    let input = b.input_array(t);
    let weight = b.input_array(u32::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let map = b.input_array(u32::as_type(&b.scope));
    let status = b.output_array(u32::as_type(&b.scope));
    let fault =
        b.output_array(cubecl::ir::StorageType::Atomic(ElemType::UInt(UIntKind::U32)).into());
    let table = b.input_array(u32::as_type(&b.scope));
    let scalars: [cubecl::ir::ManagedVariable; 48] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    expert_quant_cohort::expand(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        map.into(),
        status.into(),
        fault.into(),
        table.into(),
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
        dtype,
    );
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))
}

fn matrix_settings() -> KernelSettings {
    KernelSettings::default()
        .cube_dim(CubeDim::new_1d(64))
        .address_type(AddressType::U32)
}
type ArrayCompilation = <Array<f32> as LaunchArg>::CompilationArg;
fn register_arrays(
    launcher: &mut KernelLauncher<WgpuRuntime>,
    args: [ArrayArg<WgpuRuntime>; 7],
) -> [ArrayCompilation; 7] {
    let [input, packed, output, map, status, fault, table] = args;
    [
        <Array<Vector<f16, Const<4>>> as LaunchArg>::register(input, launcher),
        <Array<u32> as LaunchArg>::register(packed, launcher),
        <Array<f32> as LaunchArg>::register(output, launcher),
        <Array<u32> as LaunchArg>::register(map, launcher),
        <Array<u32> as LaunchArg>::register(status, launcher),
        <Array<Atomic<u32>> as LaunchArg>::register(fault, launcher),
        <Array<u32> as LaunchArg>::register(table, launcher),
    ]
}
fn register_words(launcher: &mut KernelLauncher<WgpuRuntime>, words: &[u32; 48]) {
    launcher.register_scalar_raw(
        <u32 as cubecl::CubeElement>::as_bytes(words),
        <u32 as cubecl::CubeElement>::cube_type(),
    );
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 7],
    words: &[u32; 48],
    down: bool,
    tile: usize,
    dtype: u32,
) -> Result<()> {
    crate::strata_matrix::native::compatible(client, false)?;
    let padded = crate::strata_prompt::experts::cohorts::validate_words(words)?;
    let [input, weight, output, map, status, fault, table] = h;
    if !matches!(tile, 8 | 16 | 32)
        || !supported(dtype, down)
        || words[11] != tile as u32
        || words[1] != if down { 640 } else { 2560 }
        || words[2] != if down { 2560 } else { 1280 }
        || words[5] != if down { 2 } else { 0 }
        || words[3] as usize != quant_row_bytes(dtype, down)
        || input.size_in_used() < (padded * words[1] as usize * 2) as u64
        || weight.size_in_used() != crate::strata_prompt::layout::WORKING_BYTES
        || output.size_in_used()
            != if down {
                crate::strata_prompt::groups::PART_BYTES
            } else {
                crate::strata_prompt::groups::GU_BYTES
            }
        || map.size_in_used() != (crate::strata_prompt::groups::MAP_WORDS * 4) as u64
        || status.size_in_used() != (crate::strata_prompt::groups::STATUS_WORDS * 4) as u64
        || fault.size_in_used() != 4
        || table.size_in_used() != crate::strata_intel_core::PUBLIC_TABLE_BYTES
        || !role_shape(words, dtype, down)
        || [&input, &weight, &output, &map, &status, &fault, &table]
            .iter()
            .any(|v| v.stream != input.stream)
    {
        return Err(Error::InvalidBufferLength);
    }

    let all = [&input, &weight, &output, &map, &status, &fault, &table];
    for (i, a) in all.iter().enumerate() {
        for b in &all[..i] {
            if crate::strata_prompt::storage::overlaps(a, b)? {
                return Err(Error::InvalidBufferLength);
            }
        }
    }
    for v in [&input, &weight] {
        if v.size_in_used() % 8 != 0 || v.offset_start.unwrap_or(0) % 8 != 0 {
            return Err(Error::InvalidBufferLength);
        }
    }
    let ia = unsafe { ArrayArg::from_raw_parts(input.clone(), input.size_in_used() as usize / 2) };
    let wa = crate::resident_util::raw_arg(weight.clone(), weight.size_in_used() as usize / 4)?;
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let ma = crate::resident_util::raw_arg(map.clone(), map.size_in_used() as usize / 4)?;
    let sa = crate::resident_util::raw_arg(status.clone(), status.size_in_used() as usize / 4)?;
    let ta = crate::resident_util::raw_arg(table.clone(), table.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    let settings = matrix_settings();
    let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
    launcher.with_scope(|scope| scope.device_properties(client.properties()));
    let [ia, wa, oa, ma, sa, fa, ta] = register_arrays(&mut launcher, [ia, wa, oa, ma, sa, fa, ta]);
    register_words(&mut launcher, words);
    // Every scalar is registered in its genuine generated-u32 ABI; bulk
    // serialization preserves the exact seven-array metadata and48 scalars.
    let kernel = expert_quant_cohort::ExpertQuantCohort::<WgpuRuntime>::new(
        settings,
        client.clone(),
        ia,
        wa,
        oa,
        ma,
        sa,
        fa,
        ta,
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
        dtype,
    );
    unsafe {
        launcher.launch_unchecked(
            CubeCount::Static(
                (if down { 2560usize } else { 1280usize } / 64) as u32,
                1,
                words[9],
            ),
            kernel,
            client,
        );
    }
    crate::strata_prefill_profile::count(
        if down {
            "expert_quant_cohort_down_launches"
        } else {
            "expert_quant_cohort_gu_launches"
        },
        1,
    );
    Ok(())
}
