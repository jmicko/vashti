//! Preserve the original fourteen actual CubeK routed-expert acceptance cases.
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
    let actual = super::try_expert(
        client,
        handles,
        count,
        columns,
        rows,
        weight_first,
        group,
        down,
    )?;
    if !actual {
        return Err("public CubeK expert original fallback cannot prove library execution".into());
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
        "cubek-routed-expert",
        "actual_cubek_launch",
        &[
            (32, Finite),
            (33, Finite),
            (64, Finite),
            (257, Finite),
            (33, NonfiniteFirst),
            (33, NonfiniteOther),
            (33, InvalidRoute),
        ],
        launch,
    )
}
