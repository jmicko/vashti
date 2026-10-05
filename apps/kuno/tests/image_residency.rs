use kuno::{image_residency::ImageRuntime, image_runtime::ImageConfig};
use kuno_protocol::{ImageRecipe, QWEN_RECIPE};
use std::{
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::sync::{oneshot, watch};

fn recipe() -> ImageRecipe {
    ImageRecipe {
        workflow: QWEN_RECIPE.into(),
        prompt: "synthetic".into(),
        negative_prompt: String::new(),
        seed: "42".into(),
        width: 64,
        height: 64,
        steps: 1,
        cfg: 1.0,
        attention: Default::default(),
        flow_shift: Some(3.1),
        reference_images: vec![],
    }
}
fn config(directory: &Path) -> ImageConfig {
    ImageConfig {
        models: directory.into(),
        pci_address: "0000:2d:00.0".into(),
        timeout_seconds: 5,
        collections: vec!["qwen-image-2512-fp8".into()],
        engine: Default::default(),
        runtime: None,
    }
}
fn worker(directory: &Path, job: &str) -> PathBuf {
    use base64::Engine;
    let png = base64::engine::general_purpose::STANDARD
        .encode(include_bytes!("../../vashti/src/images/fixtures/64.png"));
    let path = directory.join("worker");
    std::fs::write(
        &path,
        format!(
            r#"#!/usr/bin/python3
import sys,os,struct,time,base64,signal
directory=sys.argv[2]
def frame(kind,payload=b''):
 sys.stdout.buffer.write(b'KIP1'+bytes([kind])+struct.pack('>I',len(payload))+payload)
 sys.stdout.buffer.flush()
def stop(sig,frame):
 time.sleep(.1)
 sys.exit(0)
signal.signal(signal.SIGTERM,stop)
open(directory+'/pid','w').write(str(os.getpid()))
png=base64.b64decode('{png}')
frame(4)
while True:
 header=sys.stdin.buffer.read(9)
 if not header:break
 assert len(header)==9 and header[:4]==b'KIR2'
 size=struct.unpack('>I',header[5:])[0]
 if header[4]==2:break
 assert header[4]==1
 payload=sys.stdin.buffer.read(size)
 assert len(payload)==size
 del payload
 {job}
"#
        ),
    )
    .unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    path
}
async fn generate(
    runtime: &ImageRuntime,
    executable: &Path,
    config: &ImageConfig,
    recipe: ImageRecipe,
) -> kuno::image_residency::Finished {
    let (mut send, _receive) = oneshot::channel();
    let (_stop, shutdown) = watch::channel(false);
    let (previews, _) = watch::channel(None);
    runtime
        .run(executable, config, recipe, &mut send, shutdown, previews)
        .await
}
async fn wait_for(path: &Path) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn repeated_jobs_keep_pid_and_idle_shutdown_reaps_it() {
    let directory = tempfile::tempdir().unwrap();
    let executable = worker(directory.path(), "frame(2,png)\n frame(3)");
    let runtime = ImageRuntime::default();
    let config = config(directory.path());
    let first = generate(&runtime, &executable, &config, recipe()).await;
    assert!(first.output.is_ok() && first.settled && !first.forced);
    let pid = std::fs::read_to_string(directory.path().join("pid")).unwrap();
    assert_eq!(
        runtime.resident_device().await.as_deref(),
        Some("0000:2d:00.0")
    );
    let mut next = recipe();
    next.prompt = "different synthetic prompt".into();
    next.seed = "43".into();
    let second = generate(&runtime, &executable, &config, next).await;
    assert!(second.output.is_ok() && second.settled && !second.forced);
    assert_eq!(
        pid,
        std::fs::read_to_string(directory.path().join("pid")).unwrap()
    );
    runtime.evict().await.unwrap();
    assert!(runtime.resident_device().await.is_none());
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
}

#[tokio::test]
async fn a_changed_context_reaps_the_old_worker_before_replacement() {
    let directory = tempfile::tempdir().unwrap();
    let executable = worker(directory.path(), "frame(2,png)\n frame(3)");
    let runtime = ImageRuntime::default();
    let config = config(directory.path());
    assert!(
        generate(&runtime, &executable, &config, recipe())
            .await
            .output
            .is_ok()
    );
    let pid = std::fs::read_to_string(directory.path().join("pid")).unwrap();
    let mut next = recipe();
    next.workflow = kuno_protocol::QWEN21_RECIPE.into();
    next.flow_shift = None;
    assert!(
        generate(&runtime, &executable, &config, next)
            .await
            .output
            .is_ok()
    );
    assert_ne!(
        pid,
        std::fs::read_to_string(directory.path().join("pid")).unwrap()
    );
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
    runtime.evict().await.unwrap();
}

#[tokio::test]
async fn an_idle_worker_exit_is_recovered_on_the_next_request() {
    let directory = tempfile::tempdir().unwrap();
    let executable = worker(directory.path(), "frame(2,png)\n frame(3)");
    let runtime = ImageRuntime::default();
    let config = config(directory.path());
    assert!(
        generate(&runtime, &executable, &config, recipe())
            .await
            .output
            .is_ok()
    );
    let pid = std::fs::read_to_string(directory.path().join("pid")).unwrap();
    assert_eq!(
        unsafe { libc::kill(pid.parse().unwrap(), libc::SIGTERM) },
        0
    );
    tokio::time::sleep(Duration::from_millis(200)).await;
    let result = generate(&runtime, &executable, &config, recipe()).await;
    assert!(result.output.is_ok() && result.settled && !result.forced);
    assert_ne!(
        pid,
        std::fs::read_to_string(directory.path().join("pid")).unwrap()
    );
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
    runtime.evict().await.unwrap();
}
#[tokio::test]
async fn final_alone_does_not_release_job_until_clean_acknowledgement() {
    let directory = tempfile::tempdir().unwrap();
    let executable = worker(
        directory.path(),
        "frame(2,png)\n open(directory+'/final','w').close()\n time.sleep(.4)\n frame(3)",
    );
    let config = config(directory.path());
    let task = tokio::spawn(async move {
        let runtime = ImageRuntime::default();
        let result = generate(&runtime, &executable, &config, recipe()).await;
        runtime.evict().await.unwrap();
        result
    });
    wait_for(&directory.path().join("final")).await;
    assert!(!task.is_finished());
    assert!(task.await.unwrap().output.is_ok());
}
#[tokio::test]
async fn missing_clean_and_error_jobs_cannot_remain_resident() {
    for (job, expected) in [
        ("frame(2,png)\n sys.exit(0)", "image_worker_output_failed"),
        (
            "frame(5,b'image_request_cleanup_failed')\n sys.exit(1)",
            "image_request_cleanup_failed",
        ),
        ("frame(3)", "image_worker_frame_invalid"),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let executable = worker(directory.path(), job);
        let runtime = ImageRuntime::default();
        let result = generate(&runtime, &executable, &config(directory.path()), recipe()).await;
        assert_eq!(result.output.unwrap_err(), expected);
        assert!(result.settled && !result.forced);
        assert!(runtime.resident_device().await.is_none());
        let pid = std::fs::read_to_string(directory.path().join("pid")).unwrap();
        assert!(!Path::new(&format!("/proc/{pid}")).exists());
    }
}
#[tokio::test]
async fn cancellation_holds_supervision_through_cooperative_cleanup() {
    let directory = tempfile::tempdir().unwrap();
    let executable = worker(
        directory.path(),
        "open(directory+'/active','w').close()\n time.sleep(30)",
    );
    let config = config(directory.path());
    let runtime = std::sync::Arc::new(ImageRuntime::default());
    let owned = runtime.clone();
    let (mut send, receive) = oneshot::channel();
    let (_stop, shutdown) = watch::channel(false);
    let (previews, _) = watch::channel(None);
    let task = tokio::spawn(async move {
        owned
            .run(
                &executable,
                &config,
                recipe(),
                &mut send,
                shutdown,
                previews,
            )
            .await
    });
    wait_for(&directory.path().join("active")).await;
    drop(receive);
    let result = task.await.unwrap();
    assert_eq!(result.output.unwrap_err(), "image_cancelled");
    assert!(result.settled && !result.forced);
    assert!(runtime.resident_device().await.is_none());
    let pid = std::fs::read_to_string(directory.path().join("pid")).unwrap();
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
}

#[tokio::test]
async fn cancellation_drains_a_blocked_final_write_before_reaping() {
    let directory = tempfile::tempdir().unwrap();
    let executable = worker(
        directory.path(),
        "def finish(sig,context):\n  frame(2,b'x'*(16*1024*1024))\n  sys.exit(0)\n signal.signal(signal.SIGTERM,finish)\n open(directory+'/active','w').close()\n time.sleep(30)",
    );
    let config = config(directory.path());
    let runtime = std::sync::Arc::new(ImageRuntime::default());
    let owned = runtime.clone();
    let (mut send, receive) = oneshot::channel();
    let (_stop, shutdown) = watch::channel(false);
    let (previews, _) = watch::channel(None);
    let task = tokio::spawn(async move {
        owned
            .run(
                &executable,
                &config,
                recipe(),
                &mut send,
                shutdown,
                previews,
            )
            .await
    });
    wait_for(&directory.path().join("active")).await;
    drop(receive);
    let result = tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.output.unwrap_err(), "image_cancelled");
    assert!(result.settled && !result.forced);
    assert!(runtime.resident_device().await.is_none());
}

#[tokio::test]
async fn malformed_final_after_clean_uses_idle_shutdown_instead_of_a_signal() {
    let directory = tempfile::tempdir().unwrap();
    let executable = worker(
        directory.path(),
        "def wrong_stop(sig,context):\n  open(directory+'/signalled','w').close()\n  sys.exit(0)\n signal.signal(signal.SIGTERM,wrong_stop)\n frame(2,b'bad png')\n frame(3)",
    );
    let runtime = ImageRuntime::default();
    let result = generate(&runtime, &executable, &config(directory.path()), recipe()).await;
    assert_eq!(result.output.unwrap_err(), "image_worker_output_invalid");
    assert!(result.settled && !result.forced);
    assert!(runtime.resident_device().await.is_none());
    assert!(!directory.path().join("signalled").exists());
}

#[tokio::test]
async fn cancellation_closes_stdin_even_with_a_partially_dispatched_request() {
    let directory = tempfile::tempdir().unwrap();
    let executable = worker(directory.path(), "sys.exit(0)");
    let source = std::fs::read_to_string(&executable).unwrap()
        .replace("frame(4)", "signal.signal(signal.SIGTERM,signal.SIG_IGN)\nframe(4)")
        .replace("payload=sys.stdin.buffer.read(size)",
                 "open(directory+'/partial','w').close()\n while sys.stdin.buffer.read(1):pass\n sys.exit(0)\n payload=sys.stdin.buffer.read(size)");
    std::fs::write(&executable, source).unwrap();
    let config = config(directory.path());
    let runtime = std::sync::Arc::new(ImageRuntime::default());
    let owned = runtime.clone();
    let mut request = recipe();
    request.workflow = kuno_protocol::QWEN21_EDIT_RECIPE.into();
    request.flow_shift = None;
    request.reference_images = vec!["A".repeat(4 * 1024 * 1024)];
    let (mut send, receive) = oneshot::channel();
    let (_stop, shutdown) = watch::channel(false);
    let (previews, _) = watch::channel(None);
    let task = tokio::spawn(async move {
        owned
            .run(&executable, &config, request, &mut send, shutdown, previews)
            .await
    });
    wait_for(&directory.path().join("partial")).await;
    drop(receive);
    let result = tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.output.unwrap_err(), "image_cancelled");
    assert!(result.settled && !result.forced);
    assert!(runtime.resident_device().await.is_none());
}
