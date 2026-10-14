//! MIT Strata expert_pool_dispatch_multi's distinct routing-order split.
//! Last floor(missing*141/256) missing groups use GPU staging (native default
//! 0.55); resident groups stay on GPU. Headers/inputs remain request-private.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
    strata_global_plan::Source,
    strata_global_sources::Sources,
};
use kuno_text::{Error, Result, model::BoundGraph};
use std::sync::atomic::AtomicBool;
use zeroize::{Zeroize, Zeroizing};
pub(crate) struct Job {
    pub cell: usize,
    pub routes: Zeroizing<Vec<usize>>,
}
impl Drop for Job {
    fn drop(&mut self) {
        self.cell.zeroize();
    }
}
pub(crate) struct Plan {
    pub jobs: Vec<Job>,
    pub input: Zeroizing<Vec<f32>>,
    pub gpu_groups: usize,
}
impl Drop for Plan {
    fn drop(&mut self) {
        self.gpu_groups.zeroize();
    }
}
fn reject() -> Error {
    Error::InvalidFormat("native global CPU/GPU plan")
}
#[allow(clippy::too_many_arguments)] // Separate graph, source, admission and private-owner authorities.
pub(crate) fn prepare(
    graph: BoundGraph<'_, '_>,
    arena: &PrivateArena,
    admission: &crate::executor::ResidentAdmission,
    sources: &Sources,
    layer: usize,
    count: usize,
    map_id: BufferId,
    input_id: BufferId,
    c: &AtomicBool,
) -> Result<Plan> {
    sources.validate_graph_ready(graph)?;
    let (map, input) = admission.readback_audited(c, || {
        Ok((
            u::read_prefix_u32(arena, map_id, 804)?,
            u::read_prefix_f32(arena, input_id, count * 2560)?,
        ))
    })?;
    let (rewritten, jobs, gpu_groups) = split(&map, count, |cell| {
        Ok(matches!(
            sources.plan().source(layer, cell, 0)?,
            Source::Host { .. }
        ))
    })?;
    let mut bytes = Zeroizing::new(
        rewritten
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    arena
        .client()
        .map_err(u::fault)?
        .write_private(
            arena.handle(map_id).map_err(u::fault)?.clone(),
            std::mem::take(&mut *bytes),
        )
        .map_err(u::fault)?;
    Ok(Plan {
        jobs,
        input,
        gpu_groups,
    })
}
fn split(
    map: &[u32],
    count: usize,
    mut host: impl FnMut(usize) -> Result<bool>,
) -> Result<(Zeroizing<Vec<u32>>, Vec<Job>, usize)> {
    if !(1..=8).contains(&count)
        || map.len() != 804
        || map[0] != count as u32
        || map[1] != 0
        || map[2] != count as u32
        || map[3] == 0
        || map[3] > count as u32 * 10
    {
        return Err(reject());
    }
    let n = map[3] as usize;
    let mut missing = Zeroizing::new(vec![false; n]);
    let mut nmiss = 0;
    let mut seen = Zeroizing::new([false; 80]);
    for g in 0..n {
        let cell = map[4 + g] as usize;
        let len = map[84 + g] as usize;
        if cell >= 512 || len == 0 || len > count || map[4..4 + g].contains(&(cell as u32)) {
            return Err(reject());
        }
        missing[g] = host(cell)?;
        nmiss += usize::from(missing[g]);
        let mut rows = 0u32;
        for &r in &map[164 + g * 8..164 + g * 8 + len] {
            let r = r as usize;
            if r >= count * 10 || seen[r] || rows & (1 << (r / 10)) != 0 {
                return Err(reject());
            }
            seen[r] = true;
            rows |= 1 << (r / 10);
        }
    }
    if seen[..count * 10].iter().any(|s| !*s) {
        return Err(reject());
    }
    let transfers = (nmiss * 141 / 256).min(64);
    let mut miss_rank = 0;
    let mut cpu = Zeroizing::new(vec![false; n]);
    for g in 0..n {
        if missing[g] {
            cpu[g] = miss_rank < nmiss - transfers;
            miss_rank += 1;
        }
    }
    let mut out = Zeroizing::new(vec![0u32; 8192]);
    out[..3].copy_from_slice(&map[..3]);
    let mut jobs = Vec::new();
    let mut gpu = 0;
    for tier in [false, true] {
        for g in 0..n {
            if missing[g] != tier {
                continue;
            }
            let len = map[84 + g] as usize;
            let routes = &map[164 + g * 8..164 + g * 8 + len];
            if cpu[g] {
                let mut owned = Zeroizing::new(Vec::with_capacity(len));
                for &r in routes {
                    out[1024 + r as usize] = 1;
                    owned.push(r as usize);
                }
                jobs.push(Job {
                    cell: map[4 + g] as usize,
                    routes: owned,
                });
            } else {
                out[4 + gpu] = map[4 + g];
                out[84 + gpu] = len as u32;
                out[164 + gpu * 8..164 + gpu * 8 + len].copy_from_slice(routes);
                gpu += 1;
            }
        }
    }
    out[3] = gpu as u32;
    Ok((out, jobs, gpu))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_split_uses_distinct_last_misses_and_keeps_every_rank_exactly_once() {
        for count in 1..=8 {
            for resident in 0..=count * 10 {
                let mut map = vec![0; 804];
                map[0] = count as u32;
                map[2] = count as u32;
                map[3] = (count * 10) as u32;
                for g in 0..count * 10 {
                    map[4 + g] = g as u32;
                    map[84 + g] = 1;
                    map[164 + g * 8] = g as u32;
                }
                let (out, jobs, gpu) = split(&map, count, |cell| Ok(cell >= resident)).unwrap();
                let expected = (count * 10 - resident) * 141 / 256;
                assert_eq!(gpu, resident + expected);
                let mut visits = [0; 80];
                for j in jobs {
                    for &r in j.routes.iter() {
                        visits[r] += 1;
                        assert_eq!(out[1024 + r], 1);
                    }
                }
                for g in 0..gpu {
                    for i in 0..out[84 + g] as usize {
                        visits[out[164 + g * 8 + i] as usize] += 1;
                    }
                }
                assert_eq!(&visits[..count * 10], vec![1; count * 10].as_slice());
            }
        }
    }
}
