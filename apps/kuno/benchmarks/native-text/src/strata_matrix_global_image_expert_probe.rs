//! Additional full-owner/F64 controls; every old public population is unchanged.
use super::ordered_probe::{Control, Launcher, ProbeResult};
use super::*;
use crate::private_arena::PrivateArena;
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
) -> ProbeResult<bool> {
    if !native_expert_wide::global_register::try_expert_mode(client, h, n, k, r, w, g, d)? {
        return Err("actual global expert required".into());
    }
    Ok(true)
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
    let families: [(&'static str, bool, bool, Launcher, Launcher); 2] = [
        (
            "global-register-direct-gu",
            false,
            false,
            old_direct,
            launch,
        ),
        (
            "global-register-ordered-down",
            true,
            true,
            old_ordered,
            launch,
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
                    "actual_global_register_producer",
                    launch,
                    ordered,
                );
                arena.reset()?;
                let mut report = result?;
                report["actual_global_register_banks"] = new_selected.into();
                report["operand_shared_bytes"] =
                    serde_json::json!(if new_selected { Some(0) } else { None });
                report["original_finite_consumer"] = true.into();
                reports.push(report);
            }
        }
    }
    Ok(reports)
}
