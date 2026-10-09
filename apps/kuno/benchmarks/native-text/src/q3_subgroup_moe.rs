//! Explicit original-Q3 count1 subgroup projections; no new arithmetic policy.
//! Four rows share128 physical lanes, preserving the accepted original64
//! logical accumulators and32/16/8/4/2/1 tree from fast_matvec. Every invocation
//! participates in its barriers/shuffles, including inactive groups and rows.
//! Full MAX8 owners, immutable72-slot segments, original maps/status and final
//! finite/fault/publication checks remain with the existing caller.
//! Source-only candidate: not connected or compiled/device-accepted yet.
use super::{IQ3_XXS, IQ4_NL, IQ4_XS, Q8_0, fast, grouped_moe as g};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::{Error, Result};

const SLOTS: usize = 4;
const MEMBERS: usize = SLOTS + g::MAX_GROUPS * 2;
const GROUPS: usize = 10;

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gate_up(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
) {
    let row = CUBE_POS_X as usize * fast::ROWS_PER_CUBE + fast::local_row();
    let group = CUBE_POS_Y as usize;
    let active = g::group_active(map, map_first, group, first_slot, slot_count, 1) && row < ff;
    // Avoid an inactive map read or subtracting a segment origin until the
    // original group guard proves the complete map, route and actual slot.
    let mut slot = 0_usize;
    let mut route = 0_usize;
    if active {
        slot = map[map_first + SLOTS + group] as usize - first_slot;
        route = map[map_first + MEMBERS + group * g::MAX_BATCH] as usize;
    }
    let input_base = input_first + route / g::RANKS * input_stride;
    // These exact independently accepted helpers own the unchanged64-lane
    // accumulation/reduction. Call them unconditionally for every invocation.
    let a = fast::dot(
        gate,
        input,
        (slot * ff + row) * gate_row_bytes,
        input_base,
        active,
        hidden,
        gate_dtype,
    );
    let b = fast::dot(
        up,
        input,
        (slot * ff + row) * up_row_bytes,
        input_base,
        active,
        hidden,
        up_dtype,
    );
    if active && fast::lane() == 0 {
        activations[act_first + route * act_stride + row] = g::grouped_silu(a) * b;
        if row == 0 {
            status[status_first + route] = 1;
        }
    }
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn down(
    down_bank: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
) {
    let row = CUBE_POS_X as usize * fast::ROWS_PER_CUBE + fast::local_row();
    let group = CUBE_POS_Y as usize;
    let active = g::group_active(map, map_first, group, first_slot, slot_count, 1) && row < hidden;
    let mut slot = 0_usize;
    let mut route = 0_usize;
    if active {
        slot = map[map_first + SLOTS + group] as usize - first_slot;
        route = map[map_first + MEMBERS + group * g::MAX_BATCH] as usize;
    }
    let value = fast::dot(
        down_bank,
        activations,
        (slot * hidden + row) * down_row_bytes,
        act_first + route * act_stride,
        active,
        ff,
        down_dtype,
    );
    if active && fast::lane() == 0 {
        partials[partial_first + route * partial_stride + row] = value;
        if row == 0 {
            status[status_first + g::MAX_ROUTES + route] = 1;
        }
    }
}

fn rejected() -> Error {
    Error::InvalidFormat("original Q3 count1 subgroup projection policy")
}

pub(crate) fn validate_geometry(geometry: g::Geometry, first: usize, slots: usize) -> Result<()> {
    let [gate, up, down] = geometry.projections();
    if geometry.hidden() != 2560
        || geometry.ff() != 640
        || geometry.tile() != 1
        || geometry.capacity() != 144
        || geometry.gate_policy() != g::GatePolicy::Generic
        || !matches!(gate.dtype(), IQ3_XXS | IQ4_XS)
        || up.dtype() != gate.dtype()
        || !matches!(down.dtype(), IQ4_NL | Q8_0)
        || !matches!((first, slots), (0 | 72, 72))
    {
        return Err(rejected());
    }
    geometry.validate_count(1).map_err(|_| rejected())
}

pub(crate) fn grid(geometry: g::Geometry, is_down: bool, groups: usize) -> Result<CubeCount> {
    if !(1..=GROUPS).contains(&groups) {
        return Err(rejected());
    }
    let rows = if is_down {
        geometry.hidden()
    } else {
        geometry.ff()
    };
    Ok(CubeCount::Static(
        rows.div_ceil(fast::ROWS_PER_CUBE) as u32,
        groups as u32,
        1,
    ))
}

/// Caller retains its original pending slot/wave protection and complete
/// physical disjointness/epoch proofs through all-family completion. This
/// wrapper rechecks exact metadata, streams, full spans and shard bindings;
/// it never fills weights, changes a route, reads output or publishes slots.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [Handle; 3],
    owners: [Handle; 7],
    first: usize,
    slots: usize,
    groups: usize,
) -> Result<()> {
    validate_geometry(geometry, first, slots)?;
    let gate_grid = grid(geometry, false, groups)?;
    let down_grid = grid(geometry, true, groups)?;
    let [gate, up, down_projection] = geometry.projections();
    let bank_words = [
        gate.segment_words(slots).map_err(|_| rejected())?,
        up.segment_words(slots).map_err(|_| rejected())?,
        down_projection
            .segment_words(slots)
            .map_err(|_| rejected())?,
    ];
    geometry
        .validate_shard(first, slots, bank_words)
        .map_err(|_| rejected())?;
    let words = owners
        .each_ref()
        .map(|handle| handle.size_in_used() as usize / 4);
    if banks
        .iter()
        .zip(bank_words)
        .any(|(handle, words)| handle.size_in_used() != (words * 4) as u64)
        || banks.iter().chain(&owners).any(|handle| {
            handle.stream != owners[0].stream || !handle.size_in_used().is_multiple_of(4)
        })
    {
        return Err(rejected());
    }
    spans.validate(geometry, words).map_err(|_| rejected())?;
    let arg =
        |index: usize| unsafe { ArrayArg::from_raw_parts(owners[index].clone(), words[index]) };
    // SAFETY: exact tightly packed original shard extents and full MAX8 owner
    // spans are validated above. The original bounded group guard proves every
    // slot/route before access; all invocations reach each subgroup reduction.
    unsafe {
        gate_up::launch_unchecked::<WgpuRuntime>(
            client,
            gate_grid,
            fast::cube(),
            ArrayArg::from_raw_parts(banks[0].clone(), bank_words[0]),
            ArrayArg::from_raw_parts(banks[1].clone(), bank_words[1]),
            arg(1),
            arg(0),
            arg(2),
            arg(6),
            spans.map_first(),
            spans.input_first(),
            spans.input_stride(),
            spans.act_first(),
            spans.act_stride(),
            spans.status_first(),
            first,
            slots,
            geometry.hidden(),
            geometry.ff(),
            gate.dtype(),
            up.dtype(),
            gate.row_bytes(),
            up.row_bytes(),
        );
        down::launch_unchecked::<WgpuRuntime>(
            client,
            down_grid,
            fast::cube(),
            ArrayArg::from_raw_parts(banks[2].clone(), bank_words[2]),
            arg(2),
            arg(0),
            arg(3),
            arg(6),
            spans.map_first(),
            spans.act_first(),
            spans.act_stride(),
            spans.partial_first(),
            spans.partial_stride(),
            spans.status_first(),
            first,
            slots,
            geometry.hidden(),
            geometry.ff(),
            down_projection.dtype(),
            down_projection.row_bytes(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn geometry(gate: u32, down: u32, tile: usize, capacity: usize) -> g::Geometry {
        g::Geometry::new(
            [
                g::Projection::new(gate, 2560, 640, 512).unwrap(),
                g::Projection::new(gate, 2560, 640, 512).unwrap(),
                g::Projection::new(down, 640, 2560, 512).unwrap(),
            ],
            tile,
            capacity,
        )
        .unwrap()
    }
    #[test]
    fn only_original_q3_count1_72_slot_shards_are_admitted() {
        for (gate, down) in [(IQ3_XXS, IQ4_NL), (IQ3_XXS, Q8_0), (IQ4_XS, Q8_0)] {
            let g = geometry(gate, down, 1, 144);
            for first in [0, 72] {
                validate_geometry(g, first, 72).unwrap();
            }
            for (first, slots) in [(1, 72), (0, 71), (72, 73), (144, 72), (usize::MAX, 72)] {
                assert!(validate_geometry(g, first, slots).is_err());
            }
            assert!(validate_geometry(geometry(gate, down, 4, 144), 0, 72).is_err());
            assert!(validate_geometry(geometry(gate, down, 1, 128), 0, 64).is_err());
        }
        assert!(validate_geometry(geometry(super::super::Q4_K, Q8_0, 1, 144), 0, 72).is_err());
    }
    #[test]
    fn four_row_grids_preserve_every_first_last_feature_and_group() {
        let g = geometry(IQ3_XXS, IQ4_NL, 1, 144);
        for (down, rows) in [(false, 640usize), (true, 2560)] {
            for groups in [1, 10] {
                let CubeCount::Static(x, y, z) = grid(g, down, groups).unwrap() else {
                    panic!()
                };
                assert_eq!(
                    (x as usize * fast::ROWS_PER_CUBE, y as usize, z),
                    (rows, groups, 1)
                );
            }
            for groups in [0, 11, usize::MAX] {
                assert!(grid(g, down, groups).is_err());
            }
        }
    }
}
