use kuno::reservations::*;
use std::time::Duration;

fn scope(device: &str, worker: &str) -> Scope {
    [
        Resource::Device(device.into()),
        Resource::Worker(worker.into()),
    ]
    .into()
}

#[tokio::test]
async fn independent_work_runs_but_overlap_waits_atomically() {
    let reservations = Reservations::new(8, Duration::from_millis(50));
    let first = reservations
        .acquire(scope("intel", "ollama-intel"))
        .await
        .unwrap();
    let other = reservations
        .acquire(scope("nvidia", "bonsai"))
        .await
        .unwrap();
    let mut split = scope("intel", "split");
    split.insert(Resource::Device("nvidia".into()));
    assert!(matches!(
        reservations.acquire(split).await,
        Err("resource_queue_timeout")
    ));
    drop(first);
    drop(other);
    assert!(reservations.acquire(scope("intel", "new")).await.is_ok());
}

#[tokio::test]
async fn dropping_dispatched_work_quarantines_only_affected_live_scope() {
    let r = Reservations::new(8, Duration::from_secs(1));
    let mut lease = r.acquire(scope("intel", "ollama")).await.unwrap();
    lease.dispatched().unwrap();
    drop(lease);
    assert_eq!(r.quarantined(), scope("intel", "ollama"));
    assert!(matches!(
        r.acquire(scope("intel", "image")).await,
        Err("resource_recovery_required")
    ));
    assert!(r.acquire(scope("nvidia", "bonsai")).await.is_ok());
}

#[tokio::test]
async fn durable_marker_survives_uncertainty_and_contains_no_request_data() {
    let dir = tempfile::tempdir().unwrap();
    let r = Reservations::persistent(dir.path(), 8, Duration::from_secs(1)).unwrap();
    let mut first = r.acquire(scope("intel", "ollama")).await.unwrap();
    first.dispatched().unwrap();
    let mut second = r.acquire(scope("nvidia", "bonsai")).await.unwrap();
    second.dispatched().unwrap();
    first.complete();
    drop(first);
    assert!(dir.path().join("upstream-in-flight").exists());
    drop(second);
    drop(r);
    assert_eq!(
        std::fs::read(dir.path().join("upstream-in-flight")).unwrap(),
        b""
    );
    let restarted = Reservations::persistent(dir.path(), 8, Duration::from_secs(1)).unwrap();
    assert!(matches!(
        restarted.acquire(scope("nvidia", "bonsai")).await,
        Err("resource_recovery_required")
    ));
}

#[tokio::test]
async fn completion_clears_marker_only_after_all_runtime_work_finishes() {
    let dir = tempfile::tempdir().unwrap();
    let r = Reservations::persistent(dir.path(), 8, Duration::from_secs(1)).unwrap();
    let mut first = r.acquire(scope("intel", "ollama")).await.unwrap();
    let mut second = r.acquire(scope("nvidia", "bonsai")).await.unwrap();
    first.dispatched().unwrap();
    second.dispatched().unwrap();
    first.complete();
    drop(first);
    assert!(dir.path().join("upstream-in-flight").exists());
    second.complete();
    drop(second);
    assert!(!dir.path().join("upstream-in-flight").exists());
}

#[tokio::test]
async fn cancelled_waiters_release_capacity_and_cannot_take_a_partial_scope() {
    let r = Reservations::new(1, Duration::from_secs(5));
    let held = r.acquire(scope("intel", "ollama")).await.unwrap();
    let waiting = {
        let r = r.clone();
        tokio::spawn(async move { r.acquire(scope("intel", "image")).await })
    };
    tokio::task::yield_now().await;
    assert!(matches!(
        r.acquire(scope("intel", "other")).await,
        Err("resource_queue_full")
    ));
    waiting.abort();
    let _ = waiting.await;
    drop(held);
    assert!(r.acquire(scope("intel", "image")).await.is_ok());
}

#[tokio::test]
async fn journal_failure_prevents_runtime_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let r = Reservations::persistent(dir.path(), 8, Duration::from_secs(1)).unwrap();
    let mut lease = r.acquire(scope("intel", "ollama")).await.unwrap();
    std::fs::create_dir(dir.path().join("upstream-in-flight")).unwrap();
    assert_eq!(lease.dispatched(), Err("reservation_journal_write_failed"));
    drop(lease);
    assert!(matches!(
        r.acquire(scope("nvidia", "bonsai")).await,
        Err("resource_recovery_required")
    ));
}
