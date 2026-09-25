use kuno::worker::{DeviceSelection, Worker, WorkerConfig};
use std::os::unix::fs::PermissionsExt;

static WORKERS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn fixture(mode: &str) -> (tempfile::TempDir, WorkerConfig) {
    let dir = tempfile::tempdir().unwrap();
    let executable = dir.path().join("fake-ollama");
    let source = r##"#!/usr/bin/python3
import http.server, json, os, sys, time
mode = MODE
running = []
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self, *args): pass
 def reply(self, data):
  b=json.dumps(data).encode(); self.send_response(200); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
 def do_GET(self):
  self.reply({'models':running} if self.path=='/api/ps' else {'version':'test'})
 def do_POST(self):
  global running
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if self.path=='/api/embeddings':
   running=[dict(name=body['model'],digest=('b' if mode=='digest' else 'a')*64,size=100,size_vram=(99 if mode=='spill' else 100),context_length=(2048 if mode=='context' else body['options']['num_ctx']))]
   self.reply({'embedding':[]})
  else:
   running=[]; self.reply({'done':True})
host,port=os.environ['OLLAMA_HOST'].split(':')
server=http.server.HTTPServer((host,int(port)),Handler)
if mode == 'slow':
 with open(os.path.join(os.environ['HOME'],'startup.pid'),'w') as f: f.write(str(os.getpid()))
 time.sleep(60)
print('level=INFO msg="inference compute" library=CUDA id=0 filter_id=GPU-59ee3c71-80be-a06f-8d84-4a2d172d589c pci_id=0000:04:00.0',file=sys.stderr,flush=True)
if mode == 'runner':
 child = os.fork()
 if child == 0:
  time.sleep(60); os._exit(0)
 with open(os.path.join(os.environ['HOME'],'runner.pid'),'w') as f: f.write(str(child))
server.serve_forever()
"##.replace("MODE", &format!("{mode:?}"));
    std::fs::write(&executable, source).unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let config = WorkerConfig {
        executable,
        models: dir.path().into(),
        home: dir.path().into(),
        address: listener.local_addr().unwrap(),
        device: DeviceSelection::Cuda {
            uuid: "GPU-59ee3c71-80be-a06f-8d84-4a2d172d589c".into(),
            pci_address: "0000:04:00.0".into(),
        },
        max_loaded_models: 8,
        context_length: 4096,
    };
    (dir, config)
}

#[tokio::test]
async fn load_checks_residency_identity_and_context_before_returning_a_worker() {
    let _serial = WORKERS.lock().await;
    for (mode, expected) in [
        ("spill", "worker_cpu_spill_detected"),
        ("digest", "worker_loaded_identity_mismatch"),
        ("context", "worker_context_mismatch"),
    ] {
        let (_dir, config) = fixture(mode);
        let mut worker = Worker::start(config.clone()).await.unwrap();
        assert_eq!(
            worker.load("model", &"a".repeat(64), 4096).await,
            Err(expected)
        );
        assert!(worker.verify_alive().is_err());
        assert!(std::net::TcpListener::bind(config.address).is_ok());
    }
}

#[tokio::test]
async fn successful_load_unload_and_stop_own_only_the_worker_listener() {
    let _serial = WORKERS.lock().await;
    let (_dir, config) = fixture("success");
    let unrelated = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let mut worker = Worker::start(config.clone()).await.unwrap();
    worker.load("model", &"a".repeat(64), 4096).await.unwrap();
    let state = worker.running().await.unwrap();
    assert_eq!(state.models.len(), 1);
    assert!(state.models[0].device_assignment_verified);
    worker.unload("model").await.unwrap();
    assert!(worker.running().await.unwrap().models.is_empty());
    worker.stop().await.unwrap();
    assert!(std::net::TcpListener::bind(config.address).is_ok());
    assert!(std::net::TcpListener::bind(unrelated.local_addr().unwrap()).is_err());
}

#[tokio::test]
async fn occupied_worker_port_does_not_adopt_or_stop_another_server() {
    let _serial = WORKERS.lock().await;
    let (_dir, config) = fixture("success");
    let unrelated = std::net::TcpListener::bind(config.address).unwrap();
    assert!(matches!(
        Worker::start(config.clone()).await,
        Err("worker_port_unavailable")
    ));
    assert!(std::net::TcpListener::bind(config.address).is_err());
    drop(unrelated);
}

#[tokio::test]
async fn exited_worker_keeps_group_identity_until_its_runner_is_stopped() {
    let _serial = WORKERS.lock().await;
    let (dir, config) = fixture("runner");
    let mut worker = Worker::start(config.clone()).await.unwrap();
    let runner: u32 = std::fs::read_to_string(dir.path().join("runner.pid"))
        .unwrap()
        .parse()
        .unwrap();
    let parent = worker.process_id().unwrap();
    // SAFETY: signal only the child that this fixture owns.
    unsafe {
        libc::kill(parent as i32, libc::SIGKILL);
    }
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while worker.verify_alive().is_ok() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    worker.stop().await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let stat = std::fs::read_to_string(format!("/proc/{runner}/stat"));
            if stat.as_ref().map_or(true, |s| {
                s.split_once(") ").is_some_and(|(_, s)| s.starts_with('Z'))
            }) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("runner must be stopped even after its server exits");
    assert!(std::net::TcpListener::bind(config.address).is_ok());
}

#[tokio::test]
async fn unavailable_paths_identify_the_failed_resource_without_echoing_paths() {
    let _serial = WORKERS.lock().await;
    let (dir, config) = fixture("success");
    for (field, expected) in [
        ("models", "worker_models_directory_unavailable"),
        ("home", "worker_home_directory_unavailable"),
        ("executable", "worker_executable_unavailable"),
    ] {
        let mut invalid = config.clone();
        let missing = dir.path().join("missing-private-path");
        match field {
            "models" => invalid.models = missing,
            "home" => invalid.home = missing,
            _ => invalid.executable = missing,
        }
        assert!(matches!(Worker::start(invalid).await, Err(code) if code == expected));
    }
}

#[tokio::test]
async fn cancelled_discovery_retains_the_worker_until_reconciliation_reaps_it() {
    use kuno::{
        managed::{LoadRequest, ManagedWorker},
        reservations::Reservations,
    };
    use std::time::Duration;
    let _serial = WORKERS.lock().await;
    let (dir, config) = fixture("slow");
    let journal = tempfile::tempdir().unwrap();
    let reservations = Reservations::persistent(journal.path(), 4, Duration::from_secs(1)).unwrap();
    let worker = ManagedWorker::new(
        config.clone(),
        "cancelled-startup".into(),
        1024,
        reservations.clone(),
    )
    .unwrap();
    let preparing = worker.clone();
    let task = tokio::spawn(async move {
        preparing
            .prepare(LoadRequest {
                model: "test".into(),
                digest: "a".repeat(64),
                context: 4096,
                resident_bytes: 1,
                peak_bytes: 2,
            })
            .await
    });
    tokio::time::timeout(Duration::from_secs(3), async {
        while !dir.path().join("startup.pid").exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let pid = std::fs::read_to_string(dir.path().join("startup.pid")).unwrap();
    assert!(journal.path().join("upstream-in-flight").exists());
    task.abort();
    assert!(task.await.is_err());
    assert!(!reservations.quarantined().is_empty());
    assert!(std::path::Path::new(&format!("/proc/{pid}")).exists());
    worker.recover().await.unwrap();
    assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    assert!(reservations.quarantined().is_empty());
    assert!(!journal.path().join("upstream-in-flight").exists());
    assert!(std::net::TcpListener::bind(config.address).is_ok());
}
