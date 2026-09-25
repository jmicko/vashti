use kuno::placement::*;
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

fn workload(name: &str) -> Workload {
    Workload {
        model_digest: name.into(),
        runtime_fingerprint: "backend-build-driver".into(),
        settings_fingerprint: "ctx4096-parallel1-kvf16".into(),
    }
}

fn device(id: &str, free: u64) -> Device {
    Device {
        id: id.into(),
        total_bytes: 100,
        free_bytes: Some(free),
        reserve_bytes: 10,
    }
}

fn resident(id: &str, gpu: &str, bytes: u64) -> Resident {
    Resident {
        id: id.into(),
        worker: format!("worker-{gpu}"),
        workload: workload(id),
        allocations: BTreeMap::from([(gpu.into(), bytes)]),
        managed: true,
        placement_verified: true,
        idle: true,
        last_used: 1,
    }
}

fn candidate(model: &str, gpu: &str, peak: u64) -> Candidate {
    Candidate {
        worker: format!("worker-{gpu}"),
        workload: workload(model),
        memory: vec![Requirement {
            device: gpu.into(),
            resident_bytes: peak,
            peak_bytes: peak,
        }],
        device_control_verified: true,
    }
}

fn snapshot(devices: Vec<Device>, residents: Vec<Resident>) -> Snapshot {
    Snapshot {
        sampled_at: Instant::now(),
        devices,
        residents,
        busy_devices: Default::default(),
        busy_workers: Default::default(),
    }
}

fn choose(s: &Snapshot, candidates: &[Candidate]) -> Decision {
    plan(s, candidates, Instant::now(), Duration::from_secs(1))
}

fn ready(decision: Decision) -> Plan {
    match decision {
        Decision::Ready(plan) => plan,
        other => panic!("expected ready: {other:?}"),
    }
}

#[test]
fn warm_load_uses_only_incremental_peak_and_wins_over_preferred_cold_gpu() {
    let s = snapshot(
        vec![device("intel", 25), device("nvidia", 100)],
        vec![resident("model", "intel", 60)],
    );
    let p = ready(choose(
        &s,
        &[
            candidate("model", "nvidia", 70),
            candidate("model", "intel", 70),
        ],
    ));
    assert_eq!(p.reuse.as_deref(), Some("model"));
    assert_eq!(p.incremental_bytes["intel"], 10);
    assert!(p.evict.is_empty());
}

#[test]
fn warm_load_still_needs_workspace_and_headroom() {
    let s = snapshot(
        vec![device("intel", 15)],
        vec![resident("model", "intel", 60)],
    );
    assert_eq!(
        choose(&s, &[candidate("model", "intel", 70)]),
        Decision::Unavailable(vec![Reason::InsufficientMemory])
    );
    let s = snapshot(
        vec![device("intel", 5)],
        vec![resident("model", "intel", 60)],
    );
    assert_eq!(
        choose(&s, &[candidate("model", "intel", 60)]),
        Decision::Unavailable(vec![Reason::InsufficientMemory])
    );
}

#[test]
fn fresh_load_on_other_gpu_avoids_unnecessary_eviction() {
    let s = snapshot(
        vec![device("intel", 10), device("nvidia", 100)],
        vec![resident("old", "intel", 80)],
    );
    let p = ready(choose(
        &s,
        &[
            candidate("new", "intel", 60),
            candidate("new", "nvidia", 60),
        ],
    ));
    assert!(p.evict.is_empty());
    assert_eq!(p.devices.into_iter().collect::<Vec<_>>(), ["nvidia"]);
}

#[test]
fn eviction_is_lru_and_requires_a_new_measurement() {
    let mut old = resident("old", "intel", 20);
    old.last_used = 1;
    let mut recent = resident("recent", "intel", 30);
    recent.last_used = 2;
    let s = snapshot(vec![device("intel", 30)], vec![recent, old]);
    let Decision::EvictAndRecheck(p) = choose(&s, &[candidate("new", "intel", 35)]) else {
        panic!("expected eviction")
    };
    assert_eq!(p.evict, ["old"]);
    // An unload acknowledgment is not a memory measurement. Until telemetry
    // observes the released allocation, the cold request still cannot run.
    let delayed = snapshot(
        vec![device("intel", 30)],
        vec![resident("recent", "intel", 30)],
    );
    assert!(!matches!(
        choose(&delayed, &[candidate("new", "intel", 35)]),
        Decision::Ready(_)
    ));
    let released = snapshot(
        vec![device("intel", 50)],
        vec![resident("recent", "intel", 30)],
    );
    assert!(matches!(
        choose(&released, &[candidate("new", "intel", 35)]),
        Decision::Ready(_)
    ));
}

#[test]
fn external_and_unverified_allocations_are_never_evicted_or_credited() {
    for (managed, verified) in [(false, true), (true, false), (false, false)] {
        let mut r = resident("game-or-adopted-model", "intel", 80);
        r.managed = managed;
        r.placement_verified = verified;
        r.worker = "external-or-unreconciled-worker".into();
        let s = snapshot(vec![device("intel", 10)], vec![r]);
        assert_eq!(
            choose(&s, &[candidate("new", "intel", 30)]),
            Decision::Unavailable(vec![Reason::InsufficientMemory])
        );
    }
}

#[test]
fn active_or_reserved_gpu_is_not_reused_but_independent_gpu_can_run() {
    for reservation in [false, true] {
        let mut r = resident("model", "intel", 30);
        r.idle = reservation;
        let mut s = snapshot(vec![device("intel", 60), device("nvidia", 100)], vec![r]);
        if reservation {
            s.busy_devices.insert("intel".into());
        }
        assert_eq!(
            choose(&s, &[candidate("model", "intel", 35)]),
            Decision::Unavailable(vec![Reason::Busy])
        );
        let p = ready(choose(
            &s,
            &[
                candidate("model", "intel", 35),
                candidate("model", "nvidia", 35),
            ],
        ));
        assert!(p.devices.contains("nvidia"));
    }
}

#[test]
fn split_resident_eviction_reserves_all_affected_devices() {
    let mut r = resident("split", "intel", 40);
    r.allocations.insert("nvidia".into(), 20);
    let mut s = snapshot(vec![device("intel", 30), device("nvidia", 50)], vec![r]);
    let Decision::EvictAndRecheck(p) = choose(&s, &[candidate("new", "intel", 45)]) else {
        panic!("expected eviction")
    };
    assert_eq!(
        p.devices.into_iter().collect::<Vec<_>>(),
        ["intel", "nvidia"]
    );
    s.busy_devices.insert("nvidia".into());
    assert_eq!(
        choose(&s, &[candidate("new", "intel", 45)]),
        Decision::Unavailable(vec![Reason::InsufficientMemory])
    );
}

#[test]
fn multi_gpu_candidate_must_fit_every_device() {
    let s = snapshot(vec![device("intel", 90), device("nvidia", 20)], vec![]);
    let mut c = candidate("split", "intel", 60);
    c.memory.push(Requirement {
        device: "nvidia".into(),
        resident_bytes: 20,
        peak_bytes: 20,
    });
    assert_eq!(
        choose(&s, &[c]),
        Decision::Unavailable(vec![Reason::InsufficientMemory])
    );
}

#[test]
fn changed_model_runtime_or_settings_cannot_reuse_old_load() {
    let s = snapshot(
        vec![device("intel", 30)],
        vec![resident("model", "intel", 60)],
    );
    for field in 0..4 {
        let mut c = candidate("model", "intel", 60);
        match field {
            0 => c.workload.model_digest = "updated-digest".into(),
            1 => c.workload.runtime_fingerprint = "new-driver-or-build".into(),
            2 => c.workload.settings_fingerprint = "ctx128000".into(),
            _ => c.worker = "rust-bonsai-worker".into(),
        }
        let Decision::EvictAndRecheck(p) = choose(&s, &[c]) else {
            panic!("expected cold load")
        };
        assert!(p.reuse.is_none());
    }
}

#[test]
fn telemetry_unknown_stale_or_from_future_is_not_capacity() {
    let mut s = snapshot(vec![device("intel", 100)], vec![]);
    let c = candidate("new", "intel", 20);
    s.devices[0].free_bytes = None;
    assert_eq!(
        choose(&s, std::slice::from_ref(&c)),
        Decision::Unavailable(vec![Reason::UnknownMemory])
    );
    assert_eq!(
        plan(
            &s,
            std::slice::from_ref(&c),
            s.sampled_at + Duration::from_secs(2),
            Duration::from_secs(1)
        ),
        Decision::Unavailable(vec![Reason::StaleSnapshot])
    );
    assert_eq!(
        plan(
            &s,
            &[c],
            s.sampled_at - Duration::from_secs(1),
            Duration::from_secs(1)
        ),
        Decision::Unavailable(vec![Reason::StaleSnapshot])
    );
}

#[test]
fn impossible_or_double_counted_residency_is_rejected() {
    for residents in [
        vec![resident("one", "intel", 80)],
        vec![resident("one", "intel", 40), resident("two", "intel", 40)],
    ] {
        let s = snapshot(vec![device("intel", 30)], residents);
        assert_eq!(
            choose(&s, &[candidate("new", "intel", 20)]),
            Decision::Unavailable(vec![Reason::InvalidSnapshot])
        );
    }
}

#[test]
fn malformed_candidates_and_uncontrolled_workers_cannot_be_admitted() {
    let s = snapshot(vec![device("intel", 100)], vec![]);
    let mut c = candidate("new", "intel", 20);
    c.device_control_verified = false;
    assert_eq!(
        choose(&s, std::slice::from_ref(&c)),
        Decision::Unavailable(vec![Reason::UncontrolledWorker])
    );
    c.device_control_verified = true;
    c.memory.push(c.memory[0].clone());
    assert_eq!(
        choose(&s, std::slice::from_ref(&c)),
        Decision::Unavailable(vec![Reason::InvalidCandidate])
    );
    c.memory.pop();
    c.memory[0].peak_bytes = u64::MAX;
    assert_eq!(
        choose(&s, &[c]),
        Decision::Unavailable(vec![Reason::InsufficientMemory])
    );
}

#[test]
fn four_small_residents_are_not_limited_by_idle_profiling_cap() {
    let residents = (0..4)
        .map(|i| resident(&format!("model{i}"), "intel", 10))
        .collect();
    let s = snapshot(vec![device("intel", 55)], residents);
    let p = ready(choose(&s, &[candidate("fifth", "intel", 15)]));
    assert!(p.evict.is_empty());
}

#[test]
fn unknown_memory_on_other_half_of_split_load_prevents_eviction() {
    let mut r = resident("split", "intel", 40);
    r.allocations.insert("nvidia".into(), 20);
    let mut s = snapshot(vec![device("intel", 30), device("nvidia", 50)], vec![r]);
    s.devices[1].free_bytes = None;
    assert_eq!(
        choose(&s, &[candidate("new", "intel", 45)]),
        Decision::Unavailable(vec![Reason::InsufficientMemory])
    );
}

#[test]
fn worker_transition_blocks_other_gpu_on_same_worker() {
    let mut s = snapshot(vec![device("intel", 100), device("nvidia", 100)], vec![]);
    let mut c = candidate("new", "intel", 30);
    c.worker = "shared-worker".into();
    s.busy_workers.insert("shared-worker".into());
    assert_eq!(choose(&s, &[c]), Decision::Unavailable(vec![Reason::Busy]));
    assert!(matches!(
        choose(&s, &[candidate("new", "nvidia", 30)]),
        Decision::Ready(_)
    ));
}

#[test]
fn adopted_worker_requires_reconciliation_even_if_memory_is_free() {
    let mut r = resident("existing", "intel", 10);
    r.placement_verified = false;
    let s = snapshot(vec![device("intel", 80)], vec![r]);
    assert_eq!(
        choose(&s, &[candidate("new", "intel", 20)]),
        Decision::Unavailable(vec![Reason::UnreconciledWorker])
    );
}

#[test]
fn eviction_also_reserves_other_backend_worker() {
    let mut r = resident("old", "intel", 50);
    r.worker = "prism-bonsai".into();
    let mut s = snapshot(vec![device("intel", 30)], vec![r]);
    let Decision::EvictAndRecheck(p) = choose(&s, &[candidate("new", "intel", 40)]) else {
        panic!("expected eviction")
    };
    assert_eq!(
        p.workers.into_iter().collect::<Vec<_>>(),
        ["prism-bonsai", "worker-intel"]
    );
    s.busy_workers.insert("prism-bonsai".into());
    assert_eq!(
        choose(&s, &[candidate("new", "intel", 40)]),
        Decision::Unavailable(vec![Reason::InsufficientMemory])
    );
}
