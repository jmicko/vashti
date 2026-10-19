use axum::{
    Router,
    body::{Body, to_bytes},
    extract::Request,
    http::StatusCode,
    response::Response,
    routing::any,
};
use kuno::{config::Config, gateway::Gateway, native_text};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::{net::TcpListener, task::JoinHandle};
struct Fixture {
    base: String,
    tasks: Vec<JoinHandle<()>>,
    forwards: Arc<AtomicUsize>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}
impl Fixture {
    async fn new() -> Self {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let control = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let forwards = count.clone();
        let app = Router::new().fallback(any(move |request: Request| {
            let count = count.clone();
            async move {
                match request.uri().path() {
                    "/api/tags" => axum::Json(
                        json!({"models":[{"name":"original","custom":42}],"future_metadata":true,"query_seen":request.uri().query()}),
                    )
                    .into_response(),
                    "/v1/models" => {
                        axum::Json(json!({"data":[{"id":"original","custom":42}],"object":"list"}))
                            .into_response()
                    }
                    "/api/ps" => axum::Json(json!({"models":[]})).into_response(),
                    _ => {
                        count.fetch_add(1, Ordering::SeqCst);
                        let b = to_bytes(request.into_body(), 2 * 1024 * 1024)
                            .await
                            .unwrap();
                        Response::new(Body::from(b))
                    }
                }
            }
        }));
        let config = Config {
            listen: listener.local_addr().unwrap(),
            control_listen: control.local_addr().unwrap(),
            upstream: upstream.local_addr().unwrap(),
            managed: Some(kuno::pool::PoolConfig {
                reserve_bytes: 512 * 1024 * 1024,
                workers: vec![kuno::worker::WorkerConfig {
                    executable: "/bin/false".into(),
                    models: "/tmp/native-synthetic-models".into(),
                    home: "/tmp/native-synthetic-home".into(),
                    address: "127.0.0.1:44329".parse().unwrap(),
                    device: kuno::worker::DeviceSelection::Vulkan {
                        index: 0,
                        pci_address: "0000:2d:00.0".into(),
                    },
                    max_loaded_models: 4,
                    context_length: 4096,
                }],
            }),
            native_text: Some(native_text::Config {
                model: "kuno-test:iq3_s".into(),
                executable: "/bin/false".into(),
                settings: "/nonexistent/native-settings.json".into(),
                pci_address: "0000:2d:00.0".into(),
            }),
            ..Config::default()
        };
        let base = format!("http://{}", config.listen);
        let gateway =
            Gateway::new(config, "synthetic-control-token-at-least-32-characters").unwrap();
        let tasks = vec![
            tokio::spawn(async move {
                axum::serve(upstream, app).await.unwrap();
            }),
            tokio::spawn(async move {
                axum::serve(listener, gateway.compatibility())
                    .await
                    .unwrap();
            }),
        ];
        Self {
            base,
            tasks,
            forwards,
        }
    }
}
use axum::response::IntoResponse;
#[tokio::test]
async fn native_metadata_extends_catalog_without_discarding_upstream_fields() {
    let f = Fixture::new().await;
    let c = reqwest::Client::new();
    let tags: serde_json::Value = c
        .get(format!("{}/api/tags?future=kept", f.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(tags["models"][0]["custom"], 42);
    assert_eq!(tags["future_metadata"], true);
    assert_eq!(tags["query_seen"], "future=kept");
    assert_eq!(tags["models"][1]["name"], "kuno-test:iq3_s");
    let models: serde_json::Value = c
        .get(format!("{}/v1/models", f.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["custom"], 42);
    assert_eq!(models["data"][1]["owned_by"], "kuno");
    let show: serde_json::Value = c
        .post(format!("{}/api/show", f.base))
        .json(&json!({"model":"kuno-test:iq3_s"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(show["capabilities"], json!(["completion"]));
    assert_eq!(f.forwards.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn owned_unsupported_calls_never_reach_ollama_and_opaque_bytes_still_proxy() {
    let f = Fixture::new().await;
    let c = reqwest::Client::new();
    for (path, body) in [
        (
            "/api/generate",
            json!({"model":"kuno-test:iq3_s","prompt":"synthetic","format":"json"}),
        ),
        ("/api/create", json!({"model":"kuno-test:iq3_s"})),
        (
            "/api/copy",
            json!({"source":"kuno-test:iq3_s","destination":"other"}),
        ),
        (
            "/v1/chat/completions",
            json!({"model":"kuno-test:iq3_s","messages":[]}),
        ),
    ] {
        let r = c
            .post(format!("{}{path}", f.base))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }
    assert_eq!(f.forwards.load(Ordering::SeqCst), 0);
    let content = vec![0xff; 128 * 1024];
    let r = c
        .post(format!("{}/future/opaque", f.base))
        .body(content.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(r.bytes().await.unwrap().as_ref(), content.as_slice());
    assert_eq!(f.forwards.load(Ordering::SeqCst), 1);
    let r = c
        .post(format!("{}/api/generate", f.base))
        .json(&json!({"model":"kuno-test:iq3_s","keep_alive":0}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(
        r.json::<serde_json::Value>().await.unwrap()["done_reason"],
        "unload"
    );
}
