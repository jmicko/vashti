//! Explicit compiled-public native wide-prefix/original-tail expert acceptance.
//! No request knob, forged scope, alternate map or relaxed numerical threshold.
use super::*;
#[path = "strata_routed_expert_probe.rs"]
mod shared;

fn launch(
    client: &ComputeClient<WgpuRuntime>,
    handles: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> shared::ProbeResult<bool> {
    let actual = super::native_expert_wide::try_expert(
        client,
        handles.clone(),
        count,
        columns,
        rows,
        weight_first,
        group,
        down,
    )?;
    if count
        >= if cfg!(feature = "strata-native-expert-medium") {
            128
        } else {
            256
        }
    {
        if !actual {
            return Err("public native expert wide path was not actually queued".into());
        }
    } else {
        if actual {
            return Err("public native expert original fallback unexpectedly selected wide".into());
        }
        // False must mean no launch. Exercise the real dispatcher and its
        // original full-group kernel, rather than a synthetic replacement.
        super::native::run_expert(
            client,
            handles,
            count,
            columns,
            rows,
            weight_first,
            group,
            down,
        )?;
    }
    Ok(actual)
}

pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut crate::private_arena::PrivateArena,
) -> shared::ProbeResult<Vec<serde_json::Value>> {
    use shared::Control::*;
    shared::cases(
        client,
        arena,
        "native-wide-routed-expert",
        "actual_native_wide_launch",
        &[
            (32, Finite),
            (33, Finite),
            (64, Finite),
            (256, Finite),
            (257, Finite),
            (287, Finite),
            (288, Finite),
            (383, Finite),
            (384, Finite),
            // Every member has the bad feature: both the true wide prefix and
            // the original one-member tail must preserve fault/status policy.
            (257, NonfiniteFirst),
            (257, NonfiniteOther),
            (257, InvalidRoutePrefix),
            (257, InvalidRoute),
        ],
        launch,
    )
}

#[cfg(feature = "strata-native-direct-gu")]
#[path = "strata_direct_gu_pair_probe.rs"]
mod direct_pair;
#[cfg(feature = "strata-native-direct-gu")]
pub(crate) use direct_pair::cases as direct_gu_cases;
#[cfg(feature = "strata-native-expert-medium")]
pub(crate) use direct_pair::medium_cases as medium_direct_gu_cases;

#[cfg(feature = "strata-native-expert-medium")]
pub(crate) fn medium_cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut crate::private_arena::PrivateArena,
) -> shared::ProbeResult<Vec<serde_json::Value>> {
    use shared::Control::*;
    shared::cases(
        client,
        arena,
        "native-medium-routed-expert",
        "actual_native_wide_launch",
        &[
            (128, Finite),
            (129, Finite),
            (159, Finite),
            (160, Finite),
            (191, Finite),
            (192, Finite),
            (255, Finite),
            (129, NonfiniteFirst),
            (129, NonfiniteOther),
            (129, InvalidRoutePrefix),
            (129, InvalidRoute),
        ],
        launch,
    )
}

#[cfg(feature = "strata-narrow-global")]
pub(crate) fn narrow_cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut crate::private_arena::PrivateArena,
) -> shared::ProbeResult<Vec<serde_json::Value>> {
    use shared::Control::*;
    fn narrow_launch(
        client: &ComputeClient<WgpuRuntime>,
        h: [Handle; 6],
        count: usize,
        columns: usize,
        rows: usize,
        first: usize,
        group: usize,
        down: bool,
    ) -> shared::ProbeResult<bool> {
        if !(1..32).contains(&count) {
            return Err("public narrow fixed geometry".into());
        }
        super::native::run_expert(client, h, count, columns, rows, first, group, down)?;
        Ok(true)
    }
    shared::cases(
        client,
        arena,
        "native-global-narrow-routed-expert",
        "actual_native_global_narrow_launch",
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
        narrow_launch,
    )
}
