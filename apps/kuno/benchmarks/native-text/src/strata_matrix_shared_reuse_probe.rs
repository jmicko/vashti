//! Additional full-owner/F64 controls; every old public population is unchanged.
use super::ordered_probe::{Control, Launcher, ProbeResult};
use super::*;
use crate::private_arena::PrivateArena;
fn old_checked(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    n: usize,
    k: usize,
    r: usize,
    w: usize,
    g: usize,
    d: bool,
) -> ProbeResult<bool> {
    if !native_expert_wide::try_expert_mode(client, h, n, k, r, w, g, d, false)? {
        return Err("old actual wide prefix required".into());
    }
    Ok(false)
}
fn old_direct(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    n: usize,
    k: usize,
    r: usize,
    w: usize,
    g: usize,
    d: bool,
) -> ProbeResult<bool> {
    if !native_expert_wide::try_expert_mode(client, h, n, k, r, w, g, d, true)? {
        return Err("old actual directGU prefix required".into());
    }
    Ok(false)
}
fn old_ordered(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    n: usize,
    k: usize,
    r: usize,
    w: usize,
    g: usize,
    d: bool,
) -> ProbeResult<bool> {
    if !native_expert_wide::ordered::try_expert_mode(client, h, n, k, r, w, g, d, false)? {
        return Err("old actual ordered prefix required".into());
    }
    Ok(false)
}
fn launch(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    n: usize,
    k: usize,
    r: usize,
    w: usize,
    g: usize,
    d: bool,
    direct: bool,
    ordered: bool,
) -> ProbeResult<bool> {
    if !native_expert_wide::shared_reuse::try_expert_mode(
        client, h, n, k, r, w, g, d, direct, ordered,
    )? {
        return Err("actual explicit32KiB shared reuse prefix required".into());
    }
    Ok(true)
}
fn checked(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    n: usize,
    k: usize,
    r: usize,
    w: usize,
    g: usize,
    d: bool,
) -> ProbeResult<bool> {
    launch(client, h, n, k, r, w, g, d, false, false)
}
fn direct(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    n: usize,
    k: usize,
    r: usize,
    w: usize,
    g: usize,
    d: bool,
) -> ProbeResult<bool> {
    launch(client, h, n, k, r, w, g, d, true, false)
}
fn ordered(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    n: usize,
    k: usize,
    r: usize,
    w: usize,
    g: usize,
    d: bool,
) -> ProbeResult<bool> {
    launch(client, h, n, k, r, w, g, d, false, true)
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> ProbeResult<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled()
        || arena.buffer_count() != 0
        || !std::ptr::eq(client.info(), arena.client()?.info())
    {
        return Err("reuse original empty public audit arena required".into());
    }
    use Control::*;
    let mut controls = vec![
        (256, Finite),
        (383, Finite),
        (257, NonfiniteFirst),
        (257, NonfiniteOther),
        (257, InvalidRoutePrefix),
        (257, InvalidRoute),
    ];
    #[cfg(feature = "strata-native-expert-medium")]
    controls.extend([(128, Finite), (129, Finite)]);
    let families: [(&'static str, bool, bool, Launcher, Launcher); 4] = [
        (
            "shared-reuse-checked-gu",
            false,
            false,
            old_checked,
            checked,
        ),
        ("shared-reuse-direct-gu", false, false, old_direct, direct),
        ("shared-reuse-rank-down", true, false, old_checked, checked),
        (
            "shared-reuse-ordered-down",
            true,
            true,
            old_ordered,
            ordered,
        ),
    ];
    let mut reports = Vec::new();
    for (family, down, ordered, old, new) in families {
        for (new_selected, launch) in [(false, old), (true, new)] {
            for &(count, control) in &controls {
                let result = ordered_probe::one(
                    client,
                    arena,
                    count,
                    down,
                    control,
                    family,
                    "actual_shared_reuse_producer",
                    launch,
                    ordered,
                );
                arena.reset()?;
                let mut report = result?;
                // The original dispatcher can choose its guarded K64 variant
                // on other capable devices. Do not invent its physical layout.
                report["shared_reuse_physical_bytes"] = if new_selected {
                    serde_json::json!(32768)
                } else {
                    serde_json::Value::Null
                };
                report["actual_explicit_layout_required"] = new_selected.into();
                reports.push(report);
            }
        }
    }
    Ok(reports)
}
