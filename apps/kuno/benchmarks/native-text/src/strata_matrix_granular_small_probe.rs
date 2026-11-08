// Additional complete-owner/F64 paired controls; all prior1103 are retained.
use super::*;
use crate::private_arena::PrivateArena;
use crate::strata_matrix::ordered_probe as oracle;
fn old(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    n: usize,
    k: usize,
    r: usize,
    w: usize,
    g: usize,
    d: bool,
) -> oracle::ProbeResult<bool> {
    crate::strata_matrix::local_small::try_expert_mode(client, h, n, k, r, w, g, d, true)?
        .then_some(false)
        .ok_or_else(|| "global small original producer required".into())
}
fn new(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    n: usize,
    k: usize,
    r: usize,
    w: usize,
    g: usize,
    d: bool,
) -> oracle::ProbeResult<bool> {
    try_expert(client, h, n, k, r, w, g, d)?
        .then_some(true)
        .ok_or_else(|| "global small genuine producer required".into())
}
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
    Err("global cohort fixture must select packed entry".into())
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> oracle::ProbeResult<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled()
        || arena.buffer_count() != 0
        || !std::ptr::eq(client.info(), arena.client()?.info())
    {
        return Err("global small original empty audit arena required".into());
    }
    use oracle::Control::*;
    let controls = [
        (32, Finite),
        (33, Finite),
        (63, Finite),
        (64, Finite),
        (65, Finite),
        (127, Finite),
        (128, Finite),
        (129, Finite),
        (255, Finite),
        (33, NonfiniteFirst),
        (65, NonfiniteOther),
        (129, InvalidRoute),
        (255, InvalidLocalSentinelNonfiniteFirst(16)),
        (63, InvalidLocalSentinelNonfiniteFirst(32)),
        (32, PriorFaultNonfiniteFirst),
        (129, InvalidLocalSentinelNonfiniteFirst(127)),
    ];
    let mut reports = Vec::new();
    for (count, control) in controls {
        for down in [false, true] {
            for selected in [false, true] {
                let result = oracle::one(
                    client,
                    arena,
                    count,
                    down,
                    control,
                    "small-direct-small-granular-regular",
                    "actual_small_granular_producer",
                    if selected { new } else { old },
                    true,
                );
                if let Err(ref e) = result {
                    eprintln!(
                        "public_full_walk_case_error count={count} down={down} selected={selected} control={control:?} error={e:?}"
                    );
                }
                let reset = arena.reset();
                let mut r = result?;
                reset?;
                r["actual_small_granular_producer"] = selected.into();
                r["original_gu_and_ordered_down_contract"] = true.into();
                reports.push(r);
            }
        }
    }
    let controls = [
        (9, Finite),
        (15, Finite),
        (16, Finite),
        (17, Finite),
        (31, Finite),
        (9, NonfiniteFirst),
        (15, NonfiniteOther),
        (17, NonfiniteFirst),
        (31, InvalidRoute),
        (17, InvalidRoutePrefix),
        (31, InvalidLocalSentinelNonfiniteFirst(16)),
        (15, InvalidLocalSentinelNonfiniteFirst(8)),
        (31, PriorFaultNonfiniteFirst),
    ];
    for (count, control) in controls {
        for down in [false, true] {
            for selected in [false, true] {
                let family = if selected {
                    "small-direct-small-granular-cohort"
                } else {
                    "small-direct-small-granular-cohort-reference"
                };
                let result = oracle::one(
                    client,
                    arena,
                    count,
                    down,
                    control,
                    family,
                    "actual_small_granular_producer",
                    unused,
                    true,
                );
                if let Err(ref e) = result {
                    eprintln!(
                        "public_full_walk_cohort_error count={count} down={down} selected={selected} control={control:?} error={e:?}"
                    );
                }
                let reset = arena.reset();
                let mut r = result?;
                reset?;
                r["actual_small_granular_producer"] = selected.into();
                r["original_gu_and_ordered_down_contract"] = true.into();
                reports.push(r);
            }
        }
    }
    Ok(reports)
}
