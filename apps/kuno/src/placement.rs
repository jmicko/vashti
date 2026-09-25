//! Backend-independent, read-only placement policy for managed workers.
//!
//! A plan is NOT an admission lease. An executor must reserve every device in
//! `Plan::devices` and `Plan::workers`, reconcile again, perform any evictions, and sample again
//! before loading. Freed bytes are only hypothetical until that second sample.
//! The gateway does not activate this policy until worker control is available.
use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

/// Exact identity of a load, including backend/driver build and memory-affecting
/// settings (context, parallelism, KV precision, image dimensions, etc.). A tag
/// or model file size alone is not a workload identity or a memory estimate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Workload {
    pub model_digest: String,
    pub runtime_fingerprint: String,
    pub settings_fingerprint: String,
}

#[derive(Clone, Debug)]
pub struct Device {
    pub id: String,
    pub total_bytes: u64,
    pub free_bytes: Option<u64>,
    pub reserve_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Resident {
    pub id: String,
    pub worker: String,
    pub workload: Workload,
    /// Conservative reclaimable bytes on each physical device, established by
    /// a worker adapter. Never infer these from Ollama's aggregate size_vram.
    pub allocations: BTreeMap<String, u64>,
    pub managed: bool,
    pub placement_verified: bool,
    pub idle: bool,
    /// Monotonic use order, independent of wall-clock adjustments.
    pub last_used: u64,
}

pub struct Snapshot {
    /// Capture before collecting telemetry, so collection time counts as age.
    pub sampled_at: Instant,
    pub devices: Vec<Device>,
    pub residents: Vec<Resident>,
    /// Includes in-progress loads/unloads and outstanding reservations, even
    /// when those bytes have not appeared in hardware telemetry yet.
    pub busy_devices: BTreeSet<String>,
    /// Runtime instances also serialize transitions, even across GPUs.
    pub busy_workers: BTreeSet<String>,
}

#[derive(Clone, Debug)]
pub struct Requirement {
    pub device: String,
    pub resident_bytes: u64,
    /// Total GPU peak including resident bytes, KV cache, workspaces, etc.
    pub peak_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub worker: String,
    pub workload: Workload,
    /// Each candidate represents an adapter-supported placement, not an
    /// arbitrary distribution of bytes across otherwise unrelated devices.
    pub memory: Vec<Requirement>,
    pub device_control_verified: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    pub worker: String,
    pub reuse: Option<String>,
    pub evict: Vec<String>,
    /// Also includes other devices touched by any multi-GPU eviction.
    pub devices: BTreeSet<String>,
    pub workers: BTreeSet<String>,
    /// Additional peak above the reused allocation; do not subtract resident
    /// weights twice from a hardware free-memory reading.
    pub incremental_bytes: BTreeMap<String, u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    StaleSnapshot,
    InvalidSnapshot,
    InvalidCandidate,
    UncontrolledWorker,
    UnreconciledWorker,
    UnknownMemory,
    Busy,
    InsufficientMemory,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    Ready(Plan),
    EvictAndRecheck(Plan),
    /// Reasons correspond to candidates in preference order. A busy candidate
    /// can be queued; unknown memory requires reconciliation, not a guess.
    Unavailable(Vec<Reason>),
}

/// Prefer an exact warm load, then a load requiring no eviction, then LRU
/// eviction. Preserve caller preference order within a tier. Workload costs
/// must come from the adapter/profile; this function never uses file size.
pub fn plan(
    snapshot: &Snapshot,
    candidates: &[Candidate],
    now: Instant,
    max_age: Duration,
) -> Decision {
    if now
        .checked_duration_since(snapshot.sampled_at)
        .is_none_or(|age| age > max_age)
    {
        return Decision::Unavailable(vec![Reason::StaleSnapshot; candidates.len()]);
    }
    if !valid_snapshot(snapshot) {
        return Decision::Unavailable(vec![Reason::InvalidSnapshot; candidates.len()]);
    }
    let mut best: Option<(u8, Plan)> = None;
    let mut failures = Vec::new();
    for candidate in candidates {
        match candidate_plan(snapshot, candidate) {
            Ok(plan) => {
                let rank = if !plan.evict.is_empty() {
                    2
                } else if plan.reuse.is_some() {
                    0
                } else {
                    1
                };
                if best.as_ref().is_none_or(|(old, _)| rank < *old) {
                    best = Some((rank, plan));
                }
            }
            Err(reason) => failures.push(reason),
        }
    }
    match best {
        Some((2, plan)) => Decision::EvictAndRecheck(plan),
        Some((_, plan)) => Decision::Ready(plan),
        None => Decision::Unavailable(failures),
    }
}

fn valid_workload(workload: &Workload) -> bool {
    [
        &workload.model_digest,
        &workload.runtime_fingerprint,
        &workload.settings_fingerprint,
    ]
    .iter()
    .all(|s| !s.is_empty())
}

fn valid_snapshot(snapshot: &Snapshot) -> bool {
    let mut devices = BTreeMap::new();
    for device in &snapshot.devices {
        if device.id.is_empty()
            || device.total_bytes == 0
            || device.reserve_bytes > device.total_bytes
            || device.free_bytes.is_some_and(|n| n > device.total_bytes)
            || devices.insert(&device.id, device).is_some()
        {
            return false;
        }
    }
    if snapshot
        .busy_devices
        .iter()
        .any(|id| !devices.contains_key(id))
    {
        return false;
    }
    let mut ids = BTreeSet::new();
    let mut allocated: BTreeMap<&String, u64> = BTreeMap::new();
    for resident in &snapshot.residents {
        if resident.id.is_empty()
            || resident.worker.is_empty()
            || !valid_workload(&resident.workload)
            || !ids.insert(&resident.id)
        {
            return false;
        }
        if !resident.placement_verified {
            continue;
        }
        if resident.allocations.is_empty() {
            return false;
        }
        for (id, bytes) in &resident.allocations {
            let Some(device) = devices.get(id) else {
                return false;
            };
            let total = allocated.entry(id).or_default();
            let Some(sum) = total.checked_add(*bytes) else {
                return false;
            };
            // Inconsistent or double-counted attribution cannot be reclaimed.
            let used_ceiling = device.total_bytes - device.free_bytes.unwrap_or(0);
            if *bytes == 0 || sum > used_ceiling {
                return false;
            }
            *total = sum;
        }
    }
    true
}

fn candidate_plan(snapshot: &Snapshot, candidate: &Candidate) -> Result<Plan, Reason> {
    if candidate.worker.is_empty()
        || !valid_workload(&candidate.workload)
        || candidate.memory.is_empty()
    {
        return Err(Reason::InvalidCandidate);
    }
    if !candidate.device_control_verified {
        return Err(Reason::UncontrolledWorker);
    }
    let mut requirements = BTreeMap::new();
    for requirement in &candidate.memory {
        if requirement.resident_bytes == 0
            || requirement.peak_bytes < requirement.resident_bytes
            || requirements
                .insert(&requirement.device, requirement)
                .is_some()
        {
            return Err(Reason::InvalidCandidate);
        }
    }
    let devices: BTreeMap<_, _> = snapshot.devices.iter().map(|d| (&d.id, d)).collect();
    let mut busy = snapshot.busy_devices.clone();
    let mut busy_workers = snapshot.busy_workers.clone();
    for resident in &snapshot.residents {
        if !resident.idle {
            busy.extend(resident.allocations.keys().cloned());
            busy_workers.insert(resident.worker.clone());
        }
    }
    if busy_workers.contains(&candidate.worker) || requirements.keys().any(|id| busy.contains(*id))
    {
        return Err(Reason::Busy);
    }
    if snapshot
        .residents
        .iter()
        .any(|r| r.worker == candidate.worker && (!r.managed || !r.placement_verified))
    {
        return Err(Reason::UnreconciledWorker);
    }
    let warm = snapshot.residents.iter().find(|r| {
        r.managed
            && r.placement_verified
            && r.idle
            && r.worker == candidate.worker
            && r.workload == candidate.workload
            && r.allocations.keys().eq(requirements.keys().copied())
    });
    let mut plan = Plan {
        worker: candidate.worker.clone(),
        reuse: warm.map(|r| r.id.clone()),
        evict: Vec::new(),
        devices: requirements.keys().map(|s| (*s).clone()).collect(),
        workers: BTreeSet::from([candidate.worker.clone()]),
        incremental_bytes: BTreeMap::new(),
    };
    let mut free = BTreeMap::new();
    let mut needed = BTreeMap::new();
    for (id, requirement) in requirements {
        let device = devices.get(id).ok_or(Reason::UnknownMemory)?;
        let current_free = device.free_bytes.ok_or(Reason::UnknownMemory)?;
        let existing = warm
            .and_then(|r| r.allocations.get(id))
            .copied()
            .unwrap_or(0);
        let incremental = requirement.peak_bytes.saturating_sub(existing);
        let target = incremental
            .checked_add(device.reserve_bytes)
            .ok_or(Reason::InsufficientMemory)?;
        if target > device.total_bytes {
            return Err(Reason::InsufficientMemory);
        }
        free.insert(id.clone(), current_free);
        needed.insert(id.clone(), target);
        plan.incremental_bytes.insert(id.clone(), incremental);
    }
    let fits = |free: &BTreeMap<String, u64>| needed.iter().all(|(id, n)| free[id] >= *n);
    if fits(&free) {
        return Ok(plan);
    }
    let mut evictable: Vec<_> = snapshot
        .residents
        .iter()
        .filter(|r| {
            r.managed
                && r.placement_verified
                && r.idle
                && plan.reuse.as_ref() != Some(&r.id)
                && !busy_workers.contains(&r.worker)
                && !r.allocations.keys().any(|id| busy.contains(id))
                && r.allocations
                    .keys()
                    .all(|id| devices[id].free_bytes.is_some())
        })
        .collect();
    evictable.sort_by_key(|r| (r.last_used, &r.id));
    for resident in evictable {
        // Do not evict an LRU model on an unrelated/sufficiently-free GPU.
        if !resident
            .allocations
            .keys()
            .any(|id| needed.get(id).is_some_and(|target| free[id] < *target))
        {
            continue;
        }
        for (id, bytes) in &resident.allocations {
            if let Some(current) = free.get_mut(id) {
                *current = current.checked_add(*bytes).ok_or(Reason::InvalidSnapshot)?;
            }
            plan.devices.insert(id.clone());
        }
        plan.evict.push(resident.id.clone());
        plan.workers.insert(resident.worker.clone());
        if fits(&free) {
            return Ok(plan);
        }
    }
    Err(Reason::InsufficientMemory)
}
