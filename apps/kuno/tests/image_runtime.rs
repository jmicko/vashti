use kuno::{
    config::Config,
    gateway::Gateway,
    image_runtime::{self, ImageConfig},
};
use kuno_protocol::{ImageRecipe, QWEN_RECIPE};
use std::{os::unix::fs::PermissionsExt, time::Duration};
fn recipe() -> ImageRecipe {
    ImageRecipe {
        attention: Default::default(),
        workflow: QWEN_RECIPE.into(),
        prompt: "synthetic".into(),
        negative_prompt: "".into(),
        seed: "42".into(),
        width: 64,
        height: 64,
        steps: 1,
        cfg: 5.,
        flow_shift: Some(3.1),
        reference_images: Vec::new(),
    }
}
fn worker(directory: &std::path::Path, body: &str) -> std::path::PathBuf {
    let path = directory.join("worker");
    std::fs::write(&path, format!("#!/usr/bin/python3\nimport sys,os,time,json,struct,zlib\nr=json.load(sys.stdin)\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    path
}
fn config(path: &std::path::Path) -> ImageConfig {
    ImageConfig {
        models: path.into(),
        pci_address: "0000:2d:00.0".into(),
        timeout_seconds: 5,
        engine: Default::default(),
        runtime: None,
        collections: vec!["qwen-image-2512-fp8".into()],
    }
}

#[test]
fn image_engine_configuration_preserves_legacy_defaults_and_old_workflow() {
    use kuno::image_engine::{ImageEngine, SYCL_RUNTIME};
    let legacy = serde_json::json!({"models":"/models","pci_address":"0000:2d:00.0","timeout_seconds":60,
        "collections":["qwen-image-2512-fp8","qwen-image-2.1-int8"]});
    let mut image: ImageConfig = serde_json::from_value(legacy.clone()).unwrap();
    image.validate().unwrap();
    assert_eq!(image.engine, ImageEngine::Native);
    assert_eq!(serde_json::to_value(&image).unwrap(), legacy);
    image.engine = ImageEngine::ManagedSycl;
    assert!(image.validate().is_err());
    image.runtime = Some("../../other".into());
    assert!(image.validate().is_err());
    image.runtime = Some(SYCL_RUNTIME.into());
    image.validate().unwrap();
    assert!(!image.uses_sycl(kuno_protocol::QWEN_RECIPE));
    assert!(image.uses_sycl(kuno_protocol::QWEN21_RECIPE));
    assert!(image.uses_sycl(kuno_protocol::QWEN21_EDIT_RECIPE));
}
#[test]
fn every_known_collection_can_be_enabled_together_without_duplicates() {
    let mut image = config(std::path::Path::new("/models"));
    image.engine = kuno::image_engine::ImageEngine::ManagedSycl;
    image.runtime = Some(kuno::image_engine::SYCL_RUNTIME.into());
    image.collections = [
        "qwen-image-2512-fp8",
        "qwen-image-2.1-int8",
        kuno_protocol::MINIMAX_H3_COLLECTION,
        kuno_protocol::MINIMAX_H3_REFERENCE_COLLECTION,
    ]
    .map(String::from)
    .to_vec();
    image.validate().unwrap();
    assert!(image.video_enabled());
    let mut dup = image.collections.clone();
    dup[3] = dup[0].clone();
    image.collections = dup;
    assert_eq!(image.validate(), Err("image_collections_invalid"));
    image.collections.truncate(3);
    image.validate().unwrap();
    image.collections.push("unknown".into());
    assert_eq!(image.validate(), Err("image_collections_invalid"));
}
#[tokio::test]
async fn bounded_output_success_failure_and_deadline() {
    let d = tempfile::tempdir().unwrap();
    let executable = worker(
        d.path(),
        "def chunk(k,b):return struct.pack('>I',len(b))+k+b+struct.pack('>I',zlib.crc32(k+b))\nw=r['width'];h=r['height']\nb=b'\\x89PNG\\r\\n\\x1a\\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(bytes((w*3+1)*h)))+chunk(b'IEND',b'')\nsys.stdout.buffer.write(b)",
    );
    let (mut send, _receive) = tokio::sync::oneshot::channel();
    let finished =
        image_runtime::run(&executable, &config(d.path()), recipe(), &mut send, None).await;
    assert!(finished.reaped && finished.output.is_ok());
    let executable = worker(
        d.path(),
        "sys.stderr.write('untrusted-secret-content'); sys.exit(1)",
    );
    let finished =
        image_runtime::run(&executable, &config(d.path()), recipe(), &mut send, None).await;
    assert!(finished.reaped);
    assert_eq!(finished.output.unwrap_err(), "image_worker_failed");
    let executable = worker(
        d.path(),
        "sys.stdout.buffer.write(bytes(21*1024*1024)); time.sleep(30)",
    );
    let finished =
        image_runtime::run(&executable, &config(d.path()), recipe(), &mut send, None).await;
    assert!(finished.reaped);
    assert_eq!(
        finished.output.unwrap_err(),
        "image_worker_output_too_large"
    );
    let executable = worker(d.path(), "time.sleep(30)");
    let mut c = config(d.path());
    c.timeout_seconds = 1;
    let finished = image_runtime::run(&executable, &c, recipe(), &mut send, None).await;
    assert!(finished.reaped);
    assert_eq!(finished.output.unwrap_err(), "image_generation_timeout");
}
#[tokio::test]
async fn cancellation_reaps_worker_before_returning() {
    let d = tempfile::tempdir().unwrap();
    let executable = worker(
        d.path(),
        "import signal\ndef stop(sig,frame):\n time.sleep(.2)\n open(sys.argv[2]+'/terminated','w').write('graceful')\n sys.exit(0)\nsignal.signal(signal.SIGTERM,stop)\nopen(sys.argv[2]+'/pid','w').write(str(os.getpid()))\ntime.sleep(30)",
    );
    let c = config(d.path());
    let (mut send, receive) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        image_runtime::run(&executable, &c, recipe(), &mut send, None).await
    });
    tokio::time::timeout(Duration::from_secs(3), async {
        while !d.path().join("pid").exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let pid: u32 = std::fs::read_to_string(d.path().join("pid"))
        .unwrap()
        .parse()
        .unwrap();
    drop(receive);
    let finished = tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap();
    assert!(finished.reaped);
    assert!(!finished.forced);
    assert_eq!(finished.output.unwrap_err(), "image_cancelled");
    assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    assert_eq!(
        std::fs::read(d.path().join("terminated")).unwrap(),
        b"graceful"
    );
}

#[tokio::test]
async fn service_shutdown_also_reaps_an_active_image_worker() {
    let d = tempfile::tempdir().unwrap();
    let executable = worker(
        d.path(),
        "open(sys.argv[2]+'/pid','w').write(str(os.getpid()))\ntime.sleep(30)",
    );
    let c = config(d.path());
    let (mut send, _receive) = tokio::sync::oneshot::channel();
    let (stop, shutdown) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(async move {
        image_runtime::run(&executable, &c, recipe(), &mut send, Some(shutdown)).await
    });
    tokio::time::timeout(Duration::from_secs(3), async {
        while !d.path().join("pid").exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let pid: u32 = std::fs::read_to_string(d.path().join("pid"))
        .unwrap()
        .parse()
        .unwrap();
    stop.send_replace(true);
    let finished = tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap();
    assert!(finished.reaped);
    assert!(finished.forced);
    assert_eq!(finished.output.unwrap_err(), "image_cancelled");
    assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
}
#[tokio::test]
async fn device_loss_survives_cancellation_and_later_generic_diagnostics() {
    let d = tempfile::tempdir().unwrap();
    let executable = worker(
        d.path(),
        "import signal\ndef stop(sig,frame):\n sys.stderr.write('kuno: image_vulkan_device_lost\\nkuno: image_generation_failed\\n')\n sys.exit(1)\nsignal.signal(signal.SIGTERM,stop)\nopen(sys.argv[2]+'/pid','w').write(str(os.getpid()))\ntime.sleep(30)",
    );
    let c = config(d.path());
    let (mut send, receive) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        image_runtime::run(&executable, &c, recipe(), &mut send, None).await
    });
    tokio::time::timeout(Duration::from_secs(3), async {
        while !d.path().join("pid").exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    drop(receive);
    let finished = tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap();
    assert!(finished.reaped && !finished.forced);
    assert_eq!(finished.output.unwrap_err(), "image_vulkan_device_lost");
}
#[tokio::test]
async fn public_image_route_is_keyless_and_disabled_by_default() {
    let gateway = Gateway::new(
        Config::default(),
        "image-test-token-with-at-least-32-characters",
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/kuno/v1/images/generate",
        listener.local_addr().unwrap()
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, gateway.compatibility())
            .await
            .unwrap()
    });
    let client = reqwest::Client::new();
    assert_eq!(
        client
            .post(&url)
            .json(&recipe())
            .send()
            .await
            .unwrap()
            .status(),
        503
    );
    let response = client
        .post(&url)
        .bearer_auth("obsolete-or-incorrect-image-key")
        .json(&recipe())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    assert!(
        response
            .text()
            .await
            .unwrap()
            .contains("image_generation_not_configured")
    );
    task.abort();
}

#[cfg(test)]
mod attention_tests {
    use kuno_protocol::{ImageAttention, ImageRecipe, QWEN_RECIPE, QWEN21_RECIPE};

    #[test]
    fn attention_is_opt_in_and_rejects_unsupported_values_and_workflows() {
        let mut value = serde_json::json!({"workflow":QWEN21_RECIPE,"prompt":"test",
            "negative_prompt":"","seed":"1","width":512,"height":512,"steps":4,"cfg":1});
        let standard: ImageRecipe = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(standard.attention, ImageAttention::Standard);
        value["attention"] = "fast".into();
        let fast: ImageRecipe = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(fast.attention, ImageAttention::Fast);
        assert!(fast.validate().is_ok());
        value["workflow"] = QWEN_RECIPE.into();
        let unsupported: ImageRecipe = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(unsupported.validate(), Err("image_attention_unsupported"));
        value["attention"] = "unknown".into();
        assert!(serde_json::from_value::<ImageRecipe>(value).is_err());
    }
}
