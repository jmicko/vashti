//! Additive original/coherent raw-publication cohort controls. The shared
//! independent F64 oracle retains complete output/status/fault/source owners.
use super::*;
use crate::{private_arena::PrivateArena, strata_matrix::ordered_probe as oracle};

fn unused(
    _: &ComputeClient<WgpuRuntime>,
    _: [Handle; 6],
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: bool,
) -> oracle::ProbeResult<bool> {
    Err("small direct cohort fixture must select its genuine cohort entry".into())
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> oracle::ProbeResult<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled()
        || arena.buffer_count() != 0
        || !std::ptr::eq(client.info(), arena.client()?.info())
    {
        return Err("small direct cohort requires empty original audit arena".into());
    }
    use oracle::Control::*;
    let controls = [
        (8, Finite),
        (17, Finite),
        (8, NonfiniteFirst),
        (8, NonfiniteOther),
        (17, NonfiniteFirst),
        (17, NonfiniteOther),
        (8, InvalidRoute),
        (17, InvalidRoute),
        (17, InvalidRoutePrefix),
    ];
    let mut out = Vec::new();
    for selected in [false, true] {
        for down in [false, true] {
            for &(count, control) in &controls {
                let result = oracle::one(
                    client,
                    arena,
                    count,
                    down,
                    control,
                    "small-direct-cohort",
                    "actual_small_direct_cohort",
                    unused,
                    selected,
                );
                arena.reset()?;
                out.push(result?);
            }
        }
    }
    Ok(out)
}
