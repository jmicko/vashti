//! Paired real original versus small direct kernels, with the unchanged F64,
//! full-owner, original-weights, status/fault and noncontiguous-route oracle.
use super::*;
use crate::private_arena::PrivateArena;
use crate::strata_matrix::ordered_probe as shared;

fn original(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> shared::ProbeResult<bool> {
    super::super::run_expert(client, h, count, columns, rows, weight_first, group, down)?;
    Ok(false)
}
fn selected(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> shared::ProbeResult<bool> {
    let actual = if count
        >= if cfg!(feature = "strata-native-expert-medium") {
            128
        } else {
            256
        } {
        if down {
            crate::strata_matrix::native_expert_wide::ordered::try_expert_mode_small_tail(
                client,
                h,
                count,
                columns,
                rows,
                weight_first,
                group,
                true,
                false,
            )?
        } else {
            crate::strata_matrix::native_expert_wide::try_expert_mode_small_tail(
                client,
                h,
                count,
                columns,
                rows,
                weight_first,
                group,
                false,
                true,
            )?
        }
    } else {
        try_expert(client, h, count, columns, rows, weight_first, group, down)?
    };
    if !actual {
        return Err("public small direct actual selection missing".into());
    }
    Ok(true)
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> shared::ProbeResult<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled()
        || arena.buffer_count() != 0
        || !std::ptr::eq(client.info(), arena.client()?.info())
    {
        return Err("small direct requires original empty audit arena".into());
    }
    use shared::Control::*;
    let controls = [
        (8, Finite),
        (17, Finite),
        (32, Finite),
        (33, Finite),
        (64, Finite),
        (65, Finite),
        (129, Finite),
        (257, Finite),
        (287, Finite),
        (288, Finite),
        (8, NonfiniteFirst),
        (17, NonfiniteOther),
        (32, NonfiniteFirst),
        (64, NonfiniteOther),
        (65, InvalidRoute),
        (257, InvalidRoutePrefix),
        (287, NonfiniteFirst),
    ];
    let mut reports = Vec::new();
    for (count, control) in controls {
        for down in [false, true] {
            for mode in [false, true] {
                let result = shared::one(
                    client,
                    arena,
                    count,
                    down,
                    control,
                    "small-direct-regular",
                    "actual_small_direct_producer",
                    if mode { selected } else { original },
                    mode,
                )?;
                reports.push(result);
                arena.reset()?;
            }
        }
    }
    Ok(reports)
}
