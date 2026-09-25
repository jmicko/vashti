use kuno::admission::Admission;
use std::time::Duration;

#[tokio::test]
async fn uncertainty_survives_restart_and_does_not_store_content() {
    let directory = tempfile::tempdir().unwrap();
    let admission = Admission::persistent(1, Duration::from_secs(1), directory.path()).unwrap();
    let mut lease = admission.acquire().await.unwrap();
    lease.dispatched().unwrap();
    assert!(directory.path().join("upstream-in-flight").exists());
    drop(lease);
    assert!(admission.status().recovery_required);
    drop(admission);

    let restarted = Admission::persistent(1, Duration::from_secs(1), directory.path()).unwrap();
    assert_eq!(
        restarted.acquire().await.err(),
        Some("upstream_recovery_required")
    );
    for entry in std::fs::read_dir(directory.path()).unwrap() {
        let entry = entry.unwrap();
        assert!(matches!(
            entry.file_name().to_str().unwrap(),
            "instance.lock" | "upstream-in-flight" | "host-boot.json"
        ));
        if entry.file_name() == "host-boot.json" {
            let saved: String =
                serde_json::from_slice(&std::fs::read(entry.path()).unwrap()).unwrap();
            assert_eq!(
                saved,
                std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
                    .unwrap()
                    .trim()
            );
        } else {
            assert!(std::fs::read(entry.path()).unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn completed_work_clears_marker_and_duplicate_instance_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let admission = Admission::persistent(1, Duration::from_secs(1), directory.path()).unwrap();
    assert!(Admission::persistent(1, Duration::from_secs(1), directory.path()).is_err());
    let mut lease = admission.acquire().await.unwrap();
    lease.dispatched().unwrap();
    lease.complete();
    drop(lease);
    assert!(!directory.path().join("upstream-in-flight").exists());
    drop(admission);
    let restarted = Admission::persistent(1, Duration::from_secs(1), directory.path()).unwrap();
    assert!(!restarted.status().recovery_required);
    assert!(restarted.acquire().await.is_ok());
}

#[tokio::test]
async fn a_journal_failure_prevents_dispatch() {
    let directory = tempfile::tempdir().unwrap();
    let admission = Admission::persistent(1, Duration::from_secs(1), directory.path()).unwrap();
    let mut lease = admission.acquire().await.unwrap();
    // Simulate a filesystem conflict after startup, before dispatch.
    std::fs::create_dir(directory.path().join("upstream-in-flight")).unwrap();
    assert_eq!(lease.dispatched(), Err("admission_state_write_failed"));
    drop(lease);
    assert!(admission.status().recovery_required);
}

#[cfg(unix)]
#[test]
fn state_directory_and_lock_symlinks_are_rejected() {
    use std::os::unix::fs::symlink;
    let parent = tempfile::tempdir().unwrap();
    let real = parent.path().join("real");
    std::fs::create_dir(&real).unwrap();
    let link = parent.path().join("link");
    symlink(&real, &link).unwrap();
    assert!(Admission::persistent(1, Duration::from_secs(1), &link).is_err());
    std::fs::write(parent.path().join("unrelated"), "preserve me").unwrap();
    symlink(parent.path().join("unrelated"), real.join("instance.lock")).unwrap();
    assert!(Admission::persistent(1, Duration::from_secs(1), &real).is_err());
    assert_eq!(
        std::fs::read_to_string(parent.path().join("unrelated")).unwrap(),
        "preserve me"
    );
}
