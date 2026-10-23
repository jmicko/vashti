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
    #[cfg(feature = "strata-deep-matrix")]
    if super::native_expert_wide::deep::selected(client)
        != crate::strata_worker::deep_capability::Capability::actual(client).routed_selected()
    {
        return Err("actual deep wide compiler/capacity selection disagrees".into());
    }
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
    let reports = shared::cases(
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
    )?;
    #[cfg(feature = "strata-deep-matrix")]
    let reports = report_deep_selection(client, reports, "actual_native_wide_launch", "wide")?;
    Ok(reports)
}

#[cfg(feature = "strata-deep-matrix")]
fn report_deep_selection(
    client: &ComputeClient<WgpuRuntime>,
    mut reports: Vec<serde_json::Value>,
    original_launch_field: &str,
    family: &str,
) -> shared::ProbeResult<Vec<serde_json::Value>> {
    let observed = crate::strata_worker::deep_capability::Capability::actual(client);
    for report in &mut reports {
        let actual = report[original_launch_field]
            .as_bool()
            .ok_or("public expert actual selection record missing")?;
        report[format!("actual_deep_{family}_launch")] =
            (actual && observed.routed_selected()).into();
        report[format!("actual_original_{family}_launch")] =
            (actual && !observed.routed_selected()).into();
        report["actual_deep_capability_selection_agreed"] = true.into();
    }
    Ok(reports)
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
    let reports = shared::cases(
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
    )?;
    #[cfg(feature = "strata-deep-matrix")]
    let reports = report_deep_selection(client, reports, "actual_native_wide_launch", "wide")?;
    Ok(reports)
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

#[cfg(feature = "strata-expert-macro64")]
pub(crate) fn macro64_cases(client: &ComputeClient<WgpuRuntime>,
    arena: &mut crate::private_arena::PrivateArena) -> shared::ProbeResult<Vec<serde_json::Value>> {
    use shared::Control::*;
    fn selected(client: &ComputeClient<WgpuRuntime>, handles: [Handle;6], count: usize,
        columns: usize, rows: usize, first: usize, group: usize, down: bool) -> shared::ProbeResult<bool> {
        if !super::native::macro64_geometry(count, rows, columns, first,
            client.properties().hardware.max_shared_memory_size,
            [&handles[0],&handles[1]].map(|h| (h.size_in_used(), h.offset_start.unwrap_or(0)))) {
            return Err("actual macro64 dispatcher geometry not selected".into());
        }
        #[cfg(feature = "strata-deep-matrix")]
        if super::native::deep_macro64::selected(client,count,rows,columns,first,
            [&handles[0],&handles[1]].map(|h| (h.size_in_used(),h.offset_start.unwrap_or(0))))
            != crate::strata_worker::deep_capability::Capability::actual(client).routed_selected()
        {
            return Err("actual deep macro64 compiler/capacity selection disagrees".into());
        }
        // Counts below256 cannot select the original wide prefix. These exact
        // admitted operands select the new branch in the genuine dispatcher.
        super::native::run_expert(client,handles,count,columns,rows,first,group,down)?;
        Ok(true)
    }
    let reports=shared::cases(client,arena,"native-macro64-routed-expert","actual_macro64_launch",
        &[(64,Finite),(65,Finite),(127,Finite),(128,Finite),(129,Finite),(255,Finite),
          (65,NonfiniteFirst),(65,NonfiniteOther),(65,InvalidRoute),(65,InvalidRoutePrefix)],selected)?;
    #[cfg(feature = "strata-deep-matrix")]
    let reports = report_deep_selection(client, reports, "actual_macro64_launch", "macro64")?;
    Ok(reports)
}
