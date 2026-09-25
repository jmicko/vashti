//! Explicit opt-in hardware acceptance. No inference content is written or
//! printed. Existing services are never stopped by these tests.
use kuno::worker::{Worker, WorkerConfig};
use serde_json::{Value, json};

#[tokio::test]
#[ignore = "requires KUNO_TEST_WORKER_CONFIG; starts discovery without loading models"]
async fn owned_worker_verifies_device_without_loading_models() {
    let path = std::env::var("KUNO_TEST_WORKER_CONFIG").expect("worker config path");
    let config: WorkerConfig = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut worker = Worker::start(config.clone()).await.unwrap();
    assert!(worker.running().await.unwrap().models.is_empty());
    worker.stop().await.unwrap();
    assert!(std::net::TcpListener::bind(config.address).is_ok());
}

#[tokio::test]
#[ignore = "requires KUNO_TEST_WORKER_CONFIG and an idle assigned GPU"]
async fn owned_worker_loads_two_models_verifies_residency_and_releases_them() {
    let path = std::env::var("KUNO_TEST_WORKER_CONFIG").expect("worker config path");
    let config: WorkerConfig = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let baseline = kuno::resources::snapshot(config.address).await;
    let gpu_id = format!("pci:{}", config.device.pci_address());
    let before = baseline
        .gpus
        .iter()
        .find(|g| g.id == gpu_id)
        .unwrap()
        .memory
        .as_ref()
        .unwrap()
        .free_bytes
        .unwrap();
    let mut worker = Worker::start(config.clone()).await.unwrap();
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap();
    let catalog: Value = client
        .get(format!("http://{}/api/tags", worker.address()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let models = ["lfm2.5-thinking:1.2b-bf16", "embeddinggemma:latest"];
    for (name, context) in models.into_iter().zip([4096, 2048]) {
        let m = catalog["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == name)
            .unwrap();
        worker
            .load(name, m["digest"].as_str().unwrap(), context)
            .await
            .unwrap();
    }
    let running = worker.running().await.unwrap();
    assert_eq!(running.models.len(), 2);
    assert!(running.models.iter().all(|m| m.device_assignment_verified
        && m.residency == kuno::resources::ReportedResidency::GpuReported));
    let loaded = kuno::resources::snapshot(config.address).await;
    let free_loaded = loaded
        .gpus
        .iter()
        .find(|g| g.id == gpu_id)
        .unwrap()
        .memory
        .as_ref()
        .unwrap()
        .free_bytes
        .unwrap();
    assert!(
        before > free_loaded + 1_000_000_000,
        "expected visible GPU allocation"
    );
    let response: Value = client.post(format!("http://{}/api/generate", worker.address()))
        .json(&json!({"model":models[0],"prompt":"Reply with the word ready.","think":false,"stream":false,"options":{"num_ctx":4096,"num_predict":8,"temperature":0}}))
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(response["done"], true);
    assert!(response["eval_count"].as_u64().unwrap() > 0);
    let response: Value = client.post(format!("http://{}/api/embed", worker.address()))
        .json(&json!({"model":models[1],"input":"synthetic lifecycle test","options":{"num_ctx":2048}}))
        .send().await.unwrap().json().await.unwrap();
    assert!(!response["embeddings"][0].as_array().unwrap().is_empty());
    for name in models {
        worker.unload(name).await.unwrap();
    }
    let released = tokio::time::timeout(std::time::Duration::from_secs(15), async {
        loop {
            let state = kuno::resources::snapshot(config.address).await;
            let free = state
                .gpus
                .iter()
                .find(|g| g.id == gpu_id)
                .unwrap()
                .memory
                .as_ref()
                .unwrap()
                .free_bytes
                .unwrap();
            if state.ollama.models.is_empty() && free >= before.saturating_sub(512 * 1024 * 1024) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    })
    .await;
    worker.stop().await.unwrap();
    released.expect("memory release after unload");
    assert!(std::net::TcpListener::bind(config.address).is_ok());
}

#[tokio::test]
#[ignore = "requires KUNO_TEST_WORKER_CONFIG and an idle assigned GPU"]
async fn managed_placement_evicts_rechecks_and_recovers_cancelled_jobs() {
    use kuno::{
        managed::{LoadRequest, ManagedWorker},
        reservations::Reservations,
    };
    use std::time::Duration;
    let path = std::env::var("KUNO_TEST_WORKER_CONFIG").unwrap();
    let config: WorkerConfig = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap();
    let catalog: Value = client
        .get("http://127.0.0.1:11435/api/tags")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let digest = |name: &str| {
        catalog["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == name)
            .unwrap()["digest"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let mut text = LoadRequest {
        model: "lfm2.5-thinking:1.2b-bf16".into(),
        digest: digest("lfm2.5-thinking:1.2b-bf16"),
        context: 4096,
        resident_bytes: 3 * 1024u64.pow(3),
        peak_bytes: 4 * 1024u64.pow(3),
    };
    let mut embedding = LoadRequest {
        model: "embeddinggemma:latest".into(),
        digest: digest("embeddinggemma:latest"),
        context: 2048,
        resident_bytes: 1024u64.pow(3),
        peak_bytes: 2 * 1024u64.pow(3),
    };
    let sample = kuno::resources::snapshot(config.address).await;
    let baseline_free = sample
        .gpus
        .iter()
        .find(|g| g.pci_address == config.device.pci_address())
        .unwrap()
        .memory
        .as_ref()
        .unwrap()
        .free_bytes
        .unwrap();
    let journal = tempfile::tempdir().unwrap();
    let reservations = Reservations::persistent(journal.path(), 8, Duration::from_secs(5)).unwrap();
    let managed = ManagedWorker::new(
        config.clone(),
        "live-test-worker".into(),
        512 * 1024 * 1024,
        reservations.clone(),
    )
    .unwrap();
    let job = managed.prepare(text.clone()).await.unwrap();
    job.complete();
    // An intentionally large synthetic peak requirement exercises eviction
    // without allocating that peak or forcing a driver OOM.
    embedding.peak_bytes = baseline_free - 1024u64.pow(3);
    let job = managed.prepare(embedding.clone()).await.unwrap();
    let ps: Value = client
        .get(format!("http://{}/api/ps", job.address()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ps["models"].as_array().unwrap().len(), 1);
    assert_eq!(ps["models"][0]["name"], embedding.model);
    job.complete();
    let job = managed.prepare(text.clone()).await.unwrap();
    let response: Value = client.post(format!("http://{}/api/generate", job.address()))
        .json(&json!({"model":text.model,"prompt":"Reply ready.","think":false,"stream":false,"options":{"num_ctx":text.context,"num_predict":8}}))
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(response["done"], true);
    job.complete();
    let job = managed.prepare(text.clone()).await.unwrap();
    drop(job);
    assert!(!reservations.quarantined().is_empty());
    assert!(matches!(
        managed.prepare(text.clone()).await,
        Err("resource_recovery_required")
    ));
    managed.recover().await.unwrap();
    assert!(reservations.quarantined().is_empty());
    text.peak_bytes = 4 * 1024u64.pow(3);
    managed.prepare(text).await.unwrap().complete();
    managed.stop().await.unwrap();
    assert!(std::net::TcpListener::bind(config.address).is_ok());
    assert!(!journal.path().join("upstream-in-flight").exists());
}

#[tokio::test]
#[ignore = "requires KUNO_TEST_WORKER_CONFIG and an idle assigned GPU"]
async fn native_gateway_profiles_routes_and_unloads_on_assigned_gpu() {
    use kuno::{config::Config, gateway::Gateway, pool::PoolConfig};
    use std::time::Duration;
    let path = std::env::var("KUNO_TEST_WORKER_CONFIG").unwrap();
    let worker: WorkerConfig = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut workers = vec![worker.clone()];
    if let Ok(path) = std::env::var("KUNO_TEST_SECOND_WORKER_CONFIG") {
        workers.push(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap());
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let control = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let state = tempfile::tempdir().unwrap();
    let config = Config {
        listen: listener.local_addr().unwrap(),
        control_listen: control.local_addr().unwrap(),
        upstream: "127.0.0.1:11435".parse().unwrap(),
        state_dir: state.path().into(),
        managed: Some(PoolConfig {
            workers: workers.clone(),
            reserve_bytes: 1024u64.pow(3),
        }),
        ..Config::default()
    };
    let base = format!("http://{}", config.listen);
    let gateway = Gateway::persistent(config, "test-token-at-least-thirty-two-characters").unwrap();
    let app = gateway.compatibility();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(180))
        .build()
        .unwrap();
    // The first use of each workload is isolated. Repeating the text workload
    // can then use its observed footprint alongside the embedding model.
    for name in [
        "lfm2.5-thinking:1.2b-bf16",
        "embeddinggemma:latest",
        "lfm2.5-thinking:1.2b-bf16",
    ] {
        let embedding = name.starts_with("embedding");
        let (path, body) = if embedding {
            (
                "/api/embed",
                json!({"model":name,"input":"synthetic gateway acceptance"}),
            )
        } else {
            (
                "/api/generate",
                json!({"model":name,"prompt":"Reply with ready.","stream":false,"think":false,"options":{"num_predict":8,"temperature":0}}),
            )
        };
        let response = client
            .post(format!("{base}{path}"))
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let value: Value = response.json().await.unwrap();
        // Error codes contain no inference content; successful outputs are not printed.
        assert!(
            status.is_success(),
            "gateway failure for {name}: {}",
            value["error"]
        );
        if embedding {
            assert!(!value["embeddings"][0].as_array().unwrap().is_empty());
        } else {
            assert_eq!(value["done"], true);
        }
    }
    for (path, body, streaming) in [
        (
            "/v1/completions",
            json!({"model":"lfm2.5-thinking:1.2b-bf16","prompt":"Reply with ready.","max_tokens":8,"stream":false}),
            false,
        ),
        (
            "/v1/chat/completions",
            json!({"model":"lfm2.5-thinking:1.2b-bf16","messages":[{"role":"user","content":"Reply with ready."}],"max_tokens":8,"stream":false,"options":{"num_ctx":32768}}),
            false,
        ),
        (
            "/v1/embeddings",
            json!({"model":"embeddinggemma:latest","input":"synthetic compatibility acceptance"}),
            false,
        ),
        (
            "/v1/chat/completions",
            json!({"model":"lfm2.5-thinking:1.2b-bf16","messages":[{"role":"user","content":"Reply with ready."}],"max_tokens":8,"stream":true}),
            true,
        ),
    ] {
        let response = client
            .post(format!("{base}{path}"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert!(
            response.status().is_success(),
            "compatibility response status for {path}"
        );
        if streaming {
            let bytes = response.bytes().await.unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(text.contains("data: ") && text.contains("[DONE]"));
        } else {
            let value: Value = response.json().await.unwrap();
            if path == "/v1/embeddings" {
                assert!(!value["data"][0]["embedding"].as_array().unwrap().is_empty());
            } else {
                assert!(value["usage"]["completion_tokens"].as_u64().unwrap() > 0);
            }
        }
    }
    // Abandon a live compatibility stream. Kuno must finish draining its
    // upstream before releasing admission and its GPU/worker reservation.
    let mut abandoned=client.post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model":"lfm2.5-thinking:1.2b-bf16","messages":[{"role":"user","content":"Count upward from one."}],"max_tokens":32,"stream":true}))
        .send().await.unwrap();
    assert!(abandoned.status().is_success());
    assert!(abandoned.chunk().await.unwrap().is_some());
    drop(abandoned);
    tokio::time::timeout(Duration::from_secs(30), async {
        while gateway.admission.status().active {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("abandoned stream should finish its bounded drain");
    assert!(!gateway.admission.status().recovery_required);
    let ps: Value = client
        .get(format!("{base}/api/ps"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ps["models"].as_array().unwrap().len(), 2);
    assert!(
        ps["models"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["size"] == m["size_vram"])
    );
    assert!(!gateway.admission.status().recovery_required);
    if workers.len() == 2 {
        for worker in &workers {
            let ps: Value = client
                .get(format!("http://{}/api/ps", worker.address))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(
                ps["models"].as_array().unwrap().len(),
                1,
                "cold workload should use the empty other GPU, preserving the warm model"
            );
        }
    }
    let response = client
        .post(format!("{base}/api/generate"))
        .json(&json!({"model":"lfm2.5-thinking:1.2b-bf16","keep_alive":0,"stream":false}))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let _: Value = response.json().await.unwrap();
    for worker in &workers {
        assert!(std::net::TcpListener::bind(worker.address).is_ok());
    }
    assert!(
        !state
            .path()
            .join("gpu-reservations/upstream-in-flight")
            .exists()
    );
    assert!(!state.path().join("upstream-in-flight").exists());
    assert_eq!(
        gateway
            .usage
            .snapshot()
            .models
            .iter()
            .map(|m| m.real_calls)
            .sum::<u64>(),
        8
    );
    server.abort();
}

#[tokio::test]
#[ignore = "requires KUNO_TEST_WORKER_CONFIG and an idle assigned GPU"]
async fn failed_estimate_captures_memory_then_retries_once_without_blacklisting() {
    use kuno::{
        learning::{FailureKind, Outcome},
        managed::{LoadRequest, ManagedWorker},
        reservations::Reservations,
    };
    use std::time::Duration;
    let config: WorkerConfig = serde_json::from_slice(
        &std::fs::read(std::env::var("KUNO_TEST_WORKER_CONFIG").unwrap()).unwrap(),
    )
    .unwrap();
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap();
    let catalog: Value = client
        .get("http://127.0.0.1:11435/api/tags")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let request = |name: &str, context, resident_bytes, peak_bytes| LoadRequest {
        model: name.into(),
        digest: catalog["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == name)
            .unwrap()["digest"]
            .as_str()
            .unwrap()
            .into(),
        context,
        resident_bytes,
        peak_bytes,
    };
    let worker = ManagedWorker::new(
        config.clone(),
        "failure-evidence-test".into(),
        512 * 1024 * 1024,
        Reservations::new(8, Duration::from_secs(10)),
    )
    .unwrap();
    worker
        .prepare(request(
            "lfm2.5-thinking:1.2b-bf16",
            4096,
            3 * 1024u64.pow(3),
            4 * 1024u64.pow(3),
        ))
        .await
        .unwrap()
        .complete();
    // The model fits the GPU. Only its supplied estimate is deliberately wrong,
    // causing rejection after residency verification, before useful inference.
    let result = worker
        .prepare(request("embeddinggemma:latest", 2048, 1, 1))
        .await;
    assert!(matches!(result, Err("managed_peak_estimate_exceeded")));
    let history = worker.learning();
    assert_eq!(
        history.failure_count, 2,
        "one initial attempt and one isolated retry: {history:?}"
    );
    let first = &history.failures[0];
    assert_eq!(first.kind, FailureKind::EstimateExceeded);
    assert_eq!(first.outcome, Outcome::RecoveredNeedsControlledRetest);
    assert_eq!(first.previous_residents, 1);
    assert!(first.isolated_retry);
    let before = first.before_cleanup.unwrap().free_bytes;
    let after = kuno::learning::stable(&first.after_cleanup)
        .unwrap()
        .free_bytes;
    assert!(
        after > before + 1024u64.pow(3),
        "failed-load evidence must precede model cleanup"
    );
    assert_eq!(history.failures[1].previous_residents, 0);
    assert!(!history.failures[1].isolated_retry);
    assert!(std::net::TcpListener::bind(config.address).is_ok());
    // Failure is not a blacklist. A corrected estimate can use this GPU again.
    worker
        .prepare(request(
            "embeddinggemma:latest",
            2048,
            1024u64.pow(3),
            2 * 1024u64.pow(3),
        ))
        .await
        .unwrap()
        .complete();
    // An independently owned test worker represents unknown external GPU use.
    // Kuno must leave it running and retain an inconclusive classification.
    let external_home = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let mut external_config = config.clone();
    external_config.address = port.local_addr().unwrap();
    external_config.home = external_home.path().into();
    drop(port);
    let mut external = Worker::start(external_config).await.unwrap();
    let embedding = request("embeddinggemma:latest", 2048, 1, 1);
    external
        .load(&embedding.model, &embedding.digest, embedding.context)
        .await
        .unwrap();
    let result = worker
        .prepare(request("lfm2.5-thinking:1.2b-bf16", 4096, 1, 1))
        .await;
    assert!(matches!(result, Err("managed_peak_estimate_exceeded")));
    let history = worker.learning();
    assert_eq!(
        history.failure_count, 3,
        "unexplained pressure must suppress the isolated retry: {history:?}"
    );
    let failure = history.failures.back().unwrap();
    assert_eq!(failure.outcome, Outcome::UnexplainedMemoryPressure);
    assert!(!failure.isolated_retry);
    assert_eq!(
        external.running().await.unwrap().models.len(),
        1,
        "unrelated runtime must remain alive and loaded"
    );
    external.stop().await.unwrap();
    worker.stop().await.unwrap();
}
