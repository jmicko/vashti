use kuno::usage::{Catalog, CatalogModel, Usage};

fn catalog() -> Catalog {
    Catalog {
        models: vec![CatalogModel {
            name: "qwen:latest".into(),
            digest: "a".repeat(64),
            remote_model: None,
            remote_host: None,
        }],
    }
}

#[test]
fn only_known_artifacts_are_eligible_and_expiry_does_not_delete_profiles() {
    let usage = Usage::open(None).unwrap();
    let catalog = catalog();
    usage.record(catalog.resolve("qwen"), 100);
    usage.record(catalog.resolve("qwen:latest"), 200);
    usage.record(
        catalog.resolve("private prompt supplied as model name"),
        300,
    );
    let snapshot = usage.snapshot();
    assert_eq!(snapshot.models.len(), 1);
    assert_eq!(snapshot.models[0].real_calls, 2);
    assert_eq!(snapshot.models[0].last_real_use_unix, 200);
    assert_eq!(snapshot.unattributed_calls, 1);
    assert_eq!(usage.eligible(&catalog, 200 + 7 * 86400).len(), 1);
    assert!(usage.eligible(&catalog, 201 + 7 * 86400).is_empty());
    assert!(usage.eligible(&Catalog::default(), 300).is_empty());
    assert_eq!(usage.snapshot().models.len(), 1);
}

#[test]
fn usage_survives_restart_without_retaining_unverified_request_values() {
    let directory = tempfile::tempdir().unwrap();
    let usage = Usage::open(Some(directory.path())).unwrap();
    let catalog = catalog();
    usage.record(catalog.resolve("qwen"), 1234);
    usage.record(catalog.resolve("SECRET-INFERENCE-CONTENT"), 5678);
    drop(usage);
    let restored = Usage::open(Some(directory.path())).unwrap().snapshot();
    assert_eq!(restored.models[0].last_real_use_unix, 1234);
    assert_eq!(restored.unattributed_calls, 1);
    let file = std::fs::read_to_string(directory.path().join("usage.json")).unwrap();
    assert!(!file.contains("SECRET-INFERENCE-CONTENT"));
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn corrupted_history_is_not_silently_discarded() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("usage.json"), "damaged history").unwrap();
    assert!(Usage::open(Some(directory.path())).is_err());
    assert_eq!(
        std::fs::read_to_string(directory.path().join("usage.json")).unwrap(),
        "damaged history"
    );
}
