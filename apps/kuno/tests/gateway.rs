use axum::{
    Router,
    body::{Body, to_bytes},
    extract::Request,
    http::{HeaderValue, Method, StatusCode},
    response::Response,
    routing::any,
};
use futures_util::StreamExt;
use kuno::{
    admission::Admission,
    config::Config,
    gateway::{Gateway, RequestClass, classify},
};
use std::{sync::Arc, time::Duration};
use tokio::{
    net::TcpListener,
    sync::{Notify, mpsc},
    task::JoinHandle,
};

const TOKEN: &str = "test-token-at-least-thirty-two-characters";

struct Fixture {
    gateway: Arc<Gateway>,
    base: String,
    control: String,
    client: reqwest::Client,
    tasks: Vec<JoinHandle<()>>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

impl Fixture {
    async fn new(upstream: Router, max_waiting: usize) -> Self {
        Self::configured(upstream, max_waiting, None).await
    }

    async fn configured(
        upstream: Router,
        max_waiting: usize,
        managed: Option<kuno::pool::PoolConfig>,
    ) -> Self {
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let control_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = Config {
            listen: listener.local_addr().unwrap(),
            control_listen: control_listener.local_addr().unwrap(),
            upstream: upstream_listener.local_addr().unwrap(),
            state_dir: "unused-in-memory-fixture".into(),
            activation_file: None,
            max_waiting,
            stream_idle_timeout_seconds: 1,
            response_header_timeout_seconds: 1,
            queue_timeout_seconds: 1,
            managed,
            image: None,
        };
        let base = format!("http://{}", config.listen);
        let control = format!("http://{}", config.control_listen);
        let gateway = Gateway::new(config, TOKEN).unwrap();
        let app = gateway.compatibility();
        let ctrl = gateway.control();
        let tasks = vec![
            tokio::spawn(async move {
                axum::serve(upstream_listener, upstream).await.unwrap();
            }),
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            }),
            tokio::spawn(async move {
                axum::serve(control_listener, ctrl).await.unwrap();
            }),
        ];
        Self {
            gateway,
            base,
            control,
            tasks,
            client: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap(),
        }
    }
}

async fn until(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("condition timed out");
}

#[test]
fn exact_routes_and_methods_cannot_bypass_admission() {
    for path in [
        "/api/chat",
        "/api/generate",
        "/api/embed",
        "/api/embeddings",
        "/v1/responses",
        "/v1/responses/compact",
        "/v1/audio/transcriptions",
        "/v1/messages",
        "/v1/chat/completions",
    ] {
        assert_eq!(
            classify(&Method::POST, path),
            RequestClass::Inference,
            "{path}"
        );
    }
    assert_eq!(classify(&Method::GET, "/api/ps"), RequestClass::Metadata);
    assert_eq!(classify(&Method::POST, "/api/ps"), RequestClass::Opaque);
    assert_eq!(
        classify(&Method::GET, "/api/ps/anything"),
        RequestClass::Opaque
    );
    assert_eq!(classify(&Method::POST, "/api/show"), RequestClass::Metadata);
    assert_eq!(classify(&Method::POST, "/api/delete"), RequestClass::Opaque);
    assert_eq!(
        classify(&Method::DELETE, "/api/delete"),
        RequestClass::Catalog
    );
}

#[tokio::test]
async fn forwards_opaque_bytes_query_errors_and_end_to_end_headers() {
    let upstream = Router::new().fallback(|request: Request| async move {
        assert_eq!(request.method(), Method::PATCH);
        assert_eq!(
            request.uri().path_and_query().unwrap().as_str(),
            "/future%2Fendpoint?x=%2f&a=one&a=two"
        );
        assert_eq!(request.headers()["authorization"], "Bearer upstream-secret");
        assert_eq!(request.headers()["origin"], "http://phone.local");
        assert_eq!(
            request.headers()["content-type"],
            "multipart/form-data; boundary=example"
        );
        assert!(request.headers().get("x-remove-me").is_none());
        assert_eq!(request.headers()["x-unknown"], "preserved");
        let bytes = to_bytes(request.into_body(), 1024).await.unwrap();
        let mut response = Response::new(Body::from(bytes));
        *response.status_mut() = StatusCode::IM_A_TEAPOT;
        response
            .headers_mut()
            .insert("connection", HeaderValue::from_static("x-remove-response"));
        response
            .headers_mut()
            .insert("x-remove-response", HeaderValue::from_static("gone"));
        response
            .headers_mut()
            .insert("x-unknown-response", HeaderValue::from_static("preserved"));
        response
    });
    let f = Fixture::new(upstream, 2).await;
    let payload = b"--example\r\nopaque\x00\xff\r\n--example--";
    let response = f
        .client
        .patch(format!("{}/future%2Fendpoint?x=%2f&a=one&a=two", f.base))
        .header("authorization", "Bearer upstream-secret")
        .header("origin", "http://phone.local")
        .header("content-type", "multipart/form-data; boundary=example")
        .header("connection", "x-remove-me")
        .header("x-remove-me", "gone")
        .header("x-unknown", "preserved")
        .body(payload.to_vec())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::IM_A_TEAPOT);
    assert!(response.headers().get("x-remove-response").is_none());
    assert_eq!(response.headers()["x-unknown-response"], "preserved");
    assert_eq!(response.bytes().await.unwrap().as_ref(), payload);
    until(|| f.gateway.admission.status().completed == 1).await;
    assert!(!f.gateway.admission.status().recovery_required);
}

#[tokio::test]
async fn streams_before_completion_holds_lease_and_keeps_metadata_available() {
    let finish = Arc::new(Notify::new());
    let upstream = Router::new().fallback({
        let finish = finish.clone();
        move |request: Request| {
            let finish = finish.clone();
            async move {
                if request.uri().path() == "/api/ps" {
                    return Response::new(Body::from("{\"models\":[]}"));
                }
                let stream = futures_util::stream::unfold(0, move |state| {
                    let finish = finish.clone();
                    async move {
                        match state {
                            0 => Some((Ok::<_, std::io::Error>("{\"part\":1}\n"), 1)),
                            1 => {
                                finish.notified().await;
                                Some((Ok("{\"done\":true}\n"), 2))
                            }
                            _ => None,
                        }
                    }
                });
                Response::new(Body::from_stream(stream))
            }
        }
    });
    let f = Fixture::new(upstream, 1).await;
    let response = f
        .client
        .post(format!("{}/api/chat", f.base))
        .body("{}")
        .send()
        .await
        .unwrap();
    let mut stream = response.bytes_stream();
    let chunk = tokio::time::timeout(Duration::from_millis(500), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(chunk, "{\"part\":1}\n");
    assert!(f.gateway.admission.status().active);
    assert_eq!(
        f.client
            .get(format!("{}/api/ps", f.base))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    finish.notify_one();
    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }
    until(|| !f.gateway.admission.status().active).await;
    assert_eq!(f.gateway.admission.status().completed, 1);
}

#[tokio::test]
async fn paused_gate_blocks_inference_and_unknown_routes_without_blocking_metadata() {
    let f = Fixture::new(Router::new().fallback(|| async { "upstream" }), 1).await;
    assert_eq!(
        f.client
            .get(format!("{}/kuno/v1/resources", f.control))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let pause = format!("{}/kuno/v1/admission/pause", f.control);
    assert_eq!(
        f.client.post(&pause).send().await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    assert!(!f.gateway.admission.status().paused);
    assert_eq!(
        f.client
            .post(&pause)
            .bearer_auth(TOKEN)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    for path in [
        "/api/chat",
        "/api/create",
        "/future",
        "/kuno/v1/admission/resume",
    ] {
        assert_eq!(
            f.client
                .post(format!("{}{path}", f.base))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
    assert_eq!(
        f.client
            .get(format!("{}/api/tags", f.base))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    let status: serde_json::Value = f
        .client
        .get(format!("{}/kuno/v1/status", f.control))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["mode"], "gateway_only");
    assert_eq!(status["gpu_placement_verified"], false);
    f.client
        .post(format!("{}/kuno/v1/admission/resume", f.control))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(
        f.client
            .post(format!("{}/api/chat", f.base))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "upstream"
    );
}

#[tokio::test]
async fn bounded_queue_waits_and_does_not_dispatch_rejected_requests() {
    let f = Fixture::new(Router::new().fallback(|| async { "accepted" }), 1).await;
    let mut held = f.gateway.admission.acquire().await.unwrap();
    let client = f.client.clone();
    let url = format!("{}/api/chat", f.base);
    let queued = tokio::spawn(async move { client.post(url).send().await.unwrap() });
    until(|| f.gateway.admission.status().waiting == 1).await;
    let rejected = f
        .client
        .post(format!("{}/future", f.base))
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        rejected.json::<serde_json::Value>().await.unwrap()["error"],
        "queue_full"
    );
    held.complete();
    drop(held);
    assert_eq!(queued.await.unwrap().text().await.unwrap(), "accepted");
}

#[tokio::test]
async fn uncertain_cancellation_cannot_be_cleared_by_resume() {
    let (tx, rx) = mpsc::channel::<Result<&'static str, std::io::Error>>(1);
    tx.send(Ok("data: first\n\n")).await.unwrap();
    let receiver = Arc::new(tokio::sync::Mutex::new(Some(rx)));
    let upstream = Router::new().fallback(move || {
        let receiver = receiver.clone();
        async move {
            let rx = receiver.lock().await.take().unwrap();
            Response::new(Body::from_stream(futures_util::stream::unfold(
                rx,
                |mut rx| async move { rx.recv().await.map(|v| (v, rx)) },
            )))
        }
    });
    let f = Fixture::new(upstream, 1).await;
    let mut response = f
        .client
        .post(format!("{}/v1/responses", f.base))
        .send()
        .await
        .unwrap();
    assert!(response.chunk().await.unwrap().is_some());
    drop(response);
    until(|| f.gateway.admission.status().recovery_required).await;
    // The upstream producer sees its connection close; that is still not proof
    // that a real inference engine has stopped working on the GPU.
    tokio::time::timeout(Duration::from_secs(2), tx.closed())
        .await
        .unwrap();
    f.gateway.admission.pause(false);
    assert_eq!(
        f.gateway.admission.acquire().await.err(),
        Some("upstream_recovery_required")
    );
}

#[tokio::test]
async fn upstream_header_timeout_is_not_replayed_or_echoed() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let upstream = Router::new().fallback({
        let calls = calls.clone();
        move || {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::pending::<&'static str>().await
            }
        }
    });
    let f = Fixture::new(upstream, 1).await;
    let response = f
        .client
        .post(format!("{}/api/chat?secret=never-log", f.base))
        .body("sensitive payload")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        response.text().await.unwrap(),
        "{\"error\":\"upstream_headers_timeout\"}"
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(f.gateway.admission.status().recovery_required);
}

#[tokio::test]
async fn stalled_response_body_quarantines_the_upstream() {
    let upstream = Router::new().fallback(|| async {
        let first =
            futures_util::stream::once(async { Ok::<_, std::io::Error>("data: first\n\n") });
        let stream = first.chain(futures_util::stream::pending());
        Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(stream))
            .unwrap()
    });
    let f = Fixture::new(upstream, 1).await;
    let mut response = f
        .client
        .post(format!("{}/v1/messages", f.base))
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(response.chunk().await.unwrap().unwrap(), "data: first\n\n");
    assert!(
        tokio::time::timeout(Duration::from_secs(3), response.chunk())
            .await
            .unwrap()
            .is_err()
    );
    until(|| f.gateway.admission.status().recovery_required).await;
    assert!(!f.gateway.admission.status().active);
}

#[cfg(unix)]
#[tokio::test]
async fn binary_writes_only_content_free_state_and_diagnostics() {
    use std::process::{Command, Stdio};
    struct Process(Option<std::process::Child>);
    impl Drop for Process {
        fn drop(&mut self) {
            if let Some(child) = &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = upstream_listener.local_addr().unwrap();
    let compatibility = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let control = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let directory = tempfile::tempdir().unwrap();
    let config = Config {
        listen: compatibility.local_addr().unwrap(),
        control_listen: control.local_addr().unwrap(),
        upstream,
        state_dir: directory.path().join("state"),
        activation_file: Some(directory.path().join("activation-ready")),
        ..Config::default()
    };
    let config_path = directory.path().join("config.json");
    std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    let server = tokio::spawn(async move {
        axum::serve(
            upstream_listener,
            Router::new().fallback(|request: Request| async move {
                let bytes = to_bytes(request.into_body(), 1024).await.unwrap();
                Response::new(Body::from(bytes))
            }),
        )
        .await
        .unwrap();
    });
    drop(compatibility);
    drop(control);
    let token_file = directory.path().join("control-token");
    std::fs::write(&token_file, TOKEN).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_kuno"))
        .args(["serve", "--config"])
        .arg(&config_path)
        .env_remove("KUNO_CONTROL_TOKEN")
        .env("KUNO_CONTROL_TOKEN_FILE", &token_file)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut child = Process(Some(child));
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if client
                .get(format!("http://{}/api/version", config.listen))
                .send()
                .await
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let sensitive = "synthetic-private-body-DO-NOT-PERSIST";
    let blocked = client
        .post(format!("http://{}/api/chat", config.listen))
        .body(sensitive)
        .send()
        .await
        .unwrap();
    assert_eq!(blocked.status(), StatusCode::SERVICE_UNAVAILABLE);
    let activated: serde_json::Value = client
        .post(format!(
            "http://{}/kuno/v1/admission/resume",
            config.control_listen
        ))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(activated["admission"]["paused"], false);
    std::fs::write(config.activation_file.as_ref().unwrap(), []).unwrap();
    let response = client
        .post(format!(
            "http://{}/api/chat?private_query=DO-NOT-PERSIST",
            config.listen
        ))
        .body(sensitive)
        .send()
        .await
        .unwrap();
    assert_eq!(response.text().await.unwrap(), sensitive);
    until(|| !config.state_dir.join("upstream-in-flight").exists()).await;
    let process = child.0.take().unwrap();
    // SAFETY: signal only the child process owned by this test.
    unsafe {
        libc::kill(process.id() as i32, libc::SIGTERM);
    }
    let output = process.wait_with_output().unwrap();
    server.abort();
    assert!(output.status.success());
    let output = [output.stdout, output.stderr].concat();
    let output = String::from_utf8(output).unwrap();
    assert!(!output.contains("DO-NOT-PERSIST"));
    assert!(!output.contains(TOKEN));
    let files = std::fs::read_dir(&config.state_dir)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(files.len(), 2);
    for file in files {
        match file.file_name().to_str().unwrap() {
            "instance.lock" => assert!(std::fs::read(file.path()).unwrap().is_empty()),
            "host-boot.json" => {
                let boot: String =
                    serde_json::from_slice(&std::fs::read(file.path()).unwrap()).unwrap();
                assert_eq!(
                    boot,
                    std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
                        .unwrap()
                        .trim()
                );
            }
            other => panic!("unexpected persisted file: {other}"),
        }
    }
}

#[tokio::test]
async fn upstream_redirect_is_returned_without_following_it() {
    let upstream = Router::new().fallback(|| async {
        Response::builder()
            .status(307)
            .header("location", "http://127.0.0.1:1/do-not-follow")
            .body(Body::empty())
            .unwrap()
    });
    let f = Fixture::new(upstream, 1).await;
    let response = f
        .client
        .post(format!("{}/future", f.base))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(
        response.headers()["location"],
        "http://127.0.0.1:1/do-not-follow"
    );
    response.bytes().await.unwrap();
    until(|| !f.gateway.admission.status().active).await;
    assert!(!f.gateway.admission.status().recovery_required);
}

#[tokio::test]
async fn uploads_stream_without_waiting_for_the_entire_request() {
    let received = Arc::new(Notify::new());
    let upstream = Router::new().route(
        "/future",
        any({
            let received = received.clone();
            move |request: Request| {
                let received = received.clone();
                async move {
                    let mut body = request.into_body().into_data_stream();
                    assert_eq!(body.next().await.unwrap().unwrap(), "first");
                    received.notify_one();
                    let mut rest = Vec::new();
                    while let Some(chunk) = body.next().await {
                        rest.extend(chunk.unwrap());
                    }
                    assert_eq!(rest, b"second");
                    "ok"
                }
            }
        }),
    );
    let f = Fixture::new(upstream, 1).await;
    let (tx, rx) = mpsc::channel::<Result<&'static str, std::io::Error>>(1);
    tx.send(Ok("first")).await.unwrap();
    let body = reqwest::Body::wrap_stream(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|v| (v, rx))
    }));
    let client = f.client.clone();
    let url = format!("{}/future", f.base);
    let pending = tokio::spawn(async move { client.post(url).body(body).send().await.unwrap() });
    tokio::time::timeout(Duration::from_millis(500), received.notified())
        .await
        .unwrap();
    tx.send(Ok("second")).await.unwrap();
    drop(tx);
    assert_eq!(pending.await.unwrap().text().await.unwrap(), "ok");
}

#[tokio::test]
async fn real_usage_is_catalog_verified_and_preloads_and_metadata_do_not_refresh_it() {
    let upstream = Router::new().fallback(|request: Request| async move {
        if request.uri().path() == "/api/tags" {
            return Response::new(Body::from(
                serde_json::json!({"models": [{"name":"qwen:latest","digest":"a".repeat(64)}]})
                    .to_string(),
            ));
        }
        let bytes = to_bytes(request.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap();
        // Verify observation did not change the request body.
        if !bytes.is_empty() {
            let _: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        }
        Response::new(Body::from("{\"done\":true}"))
    });
    let f = Fixture::new(upstream, 2).await;
    for body in [
        serde_json::json!({"model":"qwen","messages":[{"role":"user","content":"PRIVATE-INPUT"}]}),
        serde_json::json!({"model":"qwen","keep_alive":0}),
        serde_json::json!({"model":"qwen","prompt":"","messages":[]}),
        serde_json::json!({"model":"PRIVATE-UNKNOWN-MODEL","prompt":"PRIVATE-INPUT"}),
    ] {
        f.client
            .post(format!("{}/api/chat", f.base))
            .json(&body)
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
    }
    f.client
        .get(format!("{}/api/tags", f.base))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(
        f.client
            .get(format!("{}/kuno/v1/usage", f.control))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let response = f
        .client
        .get(format!("{}/kuno/v1/usage", f.control))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(!response.contains("PRIVATE"));
    let snapshot: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(snapshot["models"].as_array().unwrap().len(), 1);
    assert_eq!(snapshot["models"][0]["real_calls"], 1);
    assert_eq!(snapshot["unattributed_calls"], 1);
}

#[tokio::test]
async fn timed_out_and_cancelled_waiters_do_not_leak_queue_capacity() {
    let admission = Admission::new(1, Duration::from_millis(30));
    let held = admission.acquire().await.unwrap();
    assert_eq!(admission.acquire().await.err(), Some("queue_timeout"));
    assert_eq!(admission.status().waiting, 0);
    let clone = admission.clone();
    let waiting = tokio::spawn(async move { clone.acquire().await });
    until(|| admission.status().waiting == 1).await;
    waiting.abort();
    let _ = waiting.await;
    assert_eq!(admission.status().waiting, 0);
    drop(held);
    assert!(!admission.status().recovery_required);
    assert!(admission.acquire().await.is_ok());
}

#[test]
fn configuration_prevents_loops_and_remote_control() {
    let mut config = Config {
        listen: "0.0.0.0:11434".parse().unwrap(),
        ..Config::default()
    };
    assert!(config.validate().is_err());
    config.listen = "0.0.0.0:11433".parse().unwrap();
    config.control_listen = "0.0.0.0:11432".parse().unwrap();
    assert!(config.validate().is_err());
    config.control_listen = "127.0.0.1:11432".parse().unwrap();
    config.upstream = "10.0.0.1:11434".parse().unwrap();
    assert!(config.validate().is_err());
}

#[tokio::test]
async fn client_disconnects_before_or_after_headers_drain_without_poisoning_admission() {
    for before_headers in [false, true] {
        let (tx, rx) = mpsc::channel::<Result<&'static str, std::io::Error>>(2);
        tx.send(Ok("first chunk")).await.unwrap();
        let receiver = Arc::new(tokio::sync::Mutex::new(Some(rx)));
        let started = Arc::new(Notify::new());
        let release_headers = Arc::new(Notify::new());
        let upstream = Router::new().fallback({
            let started = started.clone();
            let release_headers = release_headers.clone();
            move || {
                let receiver = receiver.clone();
                let started = started.clone();
                let release_headers = release_headers.clone();
                async move {
                    let Some(rx) = receiver.lock().await.take() else {
                        return Response::new(Body::from("next request completed"));
                    };
                    started.notify_one();
                    if before_headers {
                        release_headers.notified().await;
                    }
                    Response::new(Body::from_stream(futures_util::stream::unfold(
                        rx,
                        |mut rx| async move { rx.recv().await.map(|value| (value, rx)) },
                    )))
                }
            }
        });
        let f = Fixture::new(upstream, 1).await;
        let client = f.client.clone();
        let url = format!("{}/api/generate", f.base);
        let first = tokio::spawn(async move { client.post(url).send().await.unwrap() });
        started.notified().await;
        if before_headers {
            first.abort();
            let _ = first.await;
            release_headers.notify_one();
        } else {
            let mut response = first.await.unwrap();
            assert!(response.chunk().await.unwrap().is_some());
            drop(response);
        }
        let client = f.client.clone();
        let url = format!("{}/api/generate", f.base);
        let next =
            tokio::spawn(
                async move { client.post(url).send().await.unwrap().text().await.unwrap() },
            );
        until(|| f.gateway.admission.status().waiting == 1).await;
        assert!(f.gateway.admission.status().active);
        assert!(!f.gateway.admission.status().recovery_required);
        tx.send(Ok("discard this, do not store it")).await.unwrap();
        drop(tx);
        assert_eq!(next.await.unwrap(), "next request completed");
        until(|| f.gateway.admission.status().completed == 2).await;
        assert!(!f.gateway.admission.status().recovery_required);
    }
}

#[tokio::test]
async fn managed_cloud_routes_to_catalog_without_starting_local_worker() {
    use kuno::{
        pool::PoolConfig,
        worker::{DeviceSelection, WorkerConfig},
    };
    let upstream = Router::new().fallback(|request: Request| async move {
        match request.uri().path() {
            "/api/tags" => axum::Json(serde_json::json!({"models":[{
                "name":"cloud-model:cloud", "digest":"a".repeat(64),
                "remote_model":"cloud-model", "remote_host":"https://ollama.com"
            }]}))
            .into_response(),
            "/api/chat" => {
                let bytes = to_bytes(request.into_body(), 4096).await.unwrap();
                let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["messages"][0]["content"], "transient test input");
                assert!(value.get("options").is_none());
                axum::Json(serde_json::json!({"done":true})).into_response()
            }
            _ => panic!("unexpected cloud routing endpoint"),
        }
    });
    use axum::response::IntoResponse;
    let dir = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let managed = PoolConfig {
        reserve_bytes: 1024u64.pow(3),
        workers: vec![WorkerConfig {
            executable: dir.path().join("must-not-start"),
            models: dir.path().into(),
            home: dir.path().into(),
            address: port.local_addr().unwrap(),
            device: DeviceSelection::Vulkan {
                index: 1,
                pci_address: "0000:2d:00.0".into(),
            },
            max_loaded_models: 8,
            context_length: 4096,
        }],
    };
    let fixture = Fixture::configured(upstream, 2, Some(managed)).await;
    let response = fixture.client.post(format!("{}/api/chat",fixture.base))
        .json(&serde_json::json!({"model":"cloud-model:cloud","messages":[{"role":"user","content":"transient test input"}]}))
        .send().await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.json::<serde_json::Value>().await.unwrap()["done"],
        true
    );
    assert!(!fixture.gateway.admission.status().recovery_required);
}

#[tokio::test]
async fn persistent_gpu_uncertainty_is_visible_in_status_and_cannot_be_resumed_away() {
    use kuno::{
        pool::PoolConfig,
        worker::{DeviceSelection, WorkerConfig},
    };
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("gpu-reservations")).unwrap();
    std::fs::write(dir.path().join("gpu-reservations/upstream-in-flight"), []).unwrap();
    let control = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let public = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let upstream = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let worker = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let config = Config {
        listen: public.local_addr().unwrap(),
        upstream: upstream.local_addr().unwrap(),
        control_listen: control.local_addr().unwrap(),
        state_dir: dir.path().into(),
        managed: Some(PoolConfig {
            reserve_bytes: 1024u64.pow(3),
            workers: vec![WorkerConfig {
                executable: dir.path().join("must-not-start"),
                models: dir.path().into(),
                home: dir.path().into(),
                address: worker.local_addr().unwrap(),
                device: DeviceSelection::Vulkan {
                    index: 0,
                    pci_address: "0000:01:00.0".into(),
                },
                max_loaded_models: 8,
                context_length: 4096,
            }],
        }),
        ..Config::default()
    };
    let base = format!("http://{}", config.control_listen);
    let gateway = Gateway::persistent(config, TOKEN).unwrap();
    assert!(!gateway.admission.status().recovery_required);
    let app = gateway.control();
    let server = tokio::spawn(async move {
        axum::serve(control, app).await.unwrap();
    });
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    for path in ["/kuno/v1/status", "/kuno/v1/admission/resume"] {
        let request = if path.ends_with("resume") {
            client.post(format!("{base}{path}"))
        } else {
            client.get(format!("{base}{path}"))
        };
        let response = request.bearer_auth(TOKEN).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let status: serde_json::Value = response.json().await.unwrap();
        assert_eq!(status["mode"], "managed_experimental");
        assert_eq!(status["admission"]["recovery_required"], true);
    }
    server.abort();
}

#[tokio::test]
async fn learning_diagnostics_require_auth_and_do_not_start_workers() {
    let worker = kuno::worker::WorkerConfig {
        executable: "/nonexistent/runtime".into(),
        models: "/nonexistent/models".into(),
        home: "/nonexistent/home".into(),
        address: "127.0.0.1:39001".parse().unwrap(),
        device: kuno::worker::DeviceSelection::Vulkan {
            index: 1,
            pci_address: "0000:2d:00.0".into(),
        },
        max_loaded_models: 8,
        context_length: 4096,
    };
    for managed in [
        None,
        Some(kuno::pool::PoolConfig {
            workers: vec![worker],
            reserve_bytes: 1024u64.pow(3),
        }),
    ] {
        let enabled = managed.is_some();
        let f = Fixture::configured(Router::new(), 8, managed).await;
        let url = format!("{}/kuno/v1/learning", f.control);
        assert_eq!(
            f.client.get(&url).send().await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
        let response = f.client.get(url).bearer_auth(TOKEN).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["persistent"], false);
        assert_eq!(body["blacklisting_enabled"], false);
        assert_eq!(
            body["workers"].as_array().unwrap().len(),
            usize::from(enabled)
        );
        if enabled {
            assert_eq!(body["workers"][0]["evidence"]["failure_count"], 0);
            assert!(body["workers"][0]["evidence"]["provisional_unloaded_baseline"].is_null());
        }
    }
}
