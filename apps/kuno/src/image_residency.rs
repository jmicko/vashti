//! Model-only idle residency. Active work is still protected by global admission.
//! A worker becomes reusable only after its private request-cleanup acknowledgement.
use crate::image_runtime::{ImageConfig, Output};
use kuno_protocol::{ImageAttention, ImageRecipe};
use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Mutex, oneshot, watch},
    task::JoinHandle,
};

#[derive(PartialEq, Eq)]
struct Identity {
    executable: PathBuf,
    models: PathBuf,
    pci: String,
    version21: bool,
    attention: ImageAttention,
    runtime_worker: Option<String>,
}
struct Worker {
    identity: Identity,
    child: Child,
    input: Option<ChildStdin>,
    output: ChildStdout,
    diagnostic: JoinHandle<Option<&'static str>>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.diagnostic.abort();
    }
}

#[derive(Default)]
pub struct ImageRuntime {
    worker: Mutex<Option<Worker>>,
}
pub struct Finished {
    pub output: Output,
    /// A cleanup ack proves idle, or a normal process exit proves release.
    pub settled: bool,
    pub forced: bool,
}
impl Finished {
    fn failure(code: &'static str) -> Self {
        Self {
            output: Err(code),
            settled: code != "image_worker_cleanup_uncertain",
            forced: false,
        }
    }
}
impl ImageRuntime {
    pub async fn resident_device(&self) -> Option<String> {
        let state = self.worker.lock().await;
        state.as_ref().map(|worker| worker.identity.pci.clone())
    }
    /// Caller owns global admission. No active request is interrupted by eviction.
    pub async fn evict(&self) -> Result<(), &'static str> {
        let mut state = self.worker.lock().await;
        if let Some(mut worker) = state.take() {
            let (settled, forced, _) = stop(&mut worker, false).await;
            if !settled || forced {
                return Err("image_worker_cleanup_uncertain");
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        &self,
        executable: &Path,
        config: &ImageConfig,
        recipe: ImageRecipe,
        response: &mut oneshot::Sender<Output>,
        mut shutdown: watch::Receiver<bool>,
        previews: watch::Sender<Option<kuno_image::preview::Preview>>,
    ) -> Finished {
        if let Err(code) = config.validate().and_then(|_| recipe.validate_execution()) {
            return Finished::failure(code);
        }
        if response.is_closed() || *shutdown.borrow() {
            return Finished::failure("image_cancelled");
        }
        let mut state = self.worker.lock().await;
        let runtime = if config.uses_sycl(&recipe.workflow) {
            match crate::image_engine::VerifiedRuntime::load(config.runtime.as_deref().unwrap()) {
                Ok(runtime) => Some(runtime),
                Err(code) => return Finished::failure(code),
            }
        } else {
            None
        };
        // Legacy ABI1 remains usable during upgrade, but never receives ABI2 frames.
        if runtime
            .as_ref()
            .is_some_and(|runtime| runtime.manifest.worker_abi == 1)
        {
            if let Some(mut worker) = state.take() {
                let (settled, forced, _) = stop(&mut worker, false).await;
                if !settled || forced {
                    return Finished {
                        output: Err("image_worker_cleanup_uncertain"),
                        settled,
                        forced,
                    };
                }
            }
            let result = crate::image_runtime::run_with_previews(
                executable,
                config,
                recipe,
                response,
                Some(shutdown),
                Some(previews),
            )
            .await;
            return Finished {
                output: result.output,
                settled: result.reaped,
                forced: result.forced,
            };
        }
        let identity = Identity {
            executable: executable.into(),
            models: config.models.clone(),
            pci: config.pci_address.clone(),
            version21: recipe.workflow != kuno_protocol::QWEN_RECIPE,
            attention: recipe.attention,
            runtime_worker: runtime.as_ref().and_then(|runtime| {
                runtime
                    .manifest
                    .files
                    .iter()
                    .find(|member| member.name == "worker")
                    .map(|member| member.sha256.clone())
            }),
        };
        let recipe = kuno_image::worker_protocol::TransientRecipe::from(recipe);
        let replace = state.as_mut().is_some_and(|worker| {
            worker.identity != identity || !matches!(worker.child.try_wait(), Ok(None))
        });
        if replace && let Some(mut worker) = state.take() {
            let (settled, forced, _) = stop(&mut worker, false).await;
            if !settled || forced {
                return Finished {
                    output: Err("image_worker_cleanup_uncertain"),
                    settled,
                    forced,
                };
            }
        }
        if state.is_none() {
            match spawn(identity, config, runtime.is_some()).await {
                Ok(worker) => *state = Some(worker),
                Err(code) => return Finished::failure(code),
            }
        }
        let worker = state.as_mut().unwrap();
        if response.is_closed() || *shutdown.borrow() {
            let mut worker = state.take().unwrap();
            let (settled, forced, _) = stop(&mut worker, false).await;
            return Finished {
                output: Err("image_cancelled"),
                settled,
                forced,
            };
        }
        let mut terminal_error = false;
        let mut cleanup_ack = false;
        let work = async {
            let mut payload =
                Payload(serde_json::to_vec(&*recipe).map_err(|_| "image_request_encoding_failed")?);
            if payload.0.len() > kuno_protocol::MAX_IMAGE_REQUEST_BYTES {
                return Err("image_request_too_large");
            }
            let mut header = *b"KIR2\x01\0\0\0\0";
            header[5..9].copy_from_slice(&(payload.0.len() as u32).to_be_bytes());
            worker
                .input
                .as_mut()
                .unwrap()
                .write_all(&header)
                .await
                .map_err(|_| "image_worker_input_failed")?;
            worker
                .input
                .as_mut()
                .unwrap()
                .write_all(&payload.0)
                .await
                .map_err(|_| "image_worker_input_failed")?;
            worker
                .input
                .as_mut()
                .unwrap()
                .flush()
                .await
                .map_err(|_| "image_worker_input_failed")?;
            payload.clear();
            let png = read_result(
                &mut worker.output,
                &previews,
                recipe.steps,
                &mut terminal_error,
                &mut cleanup_ack,
            )
            .await?;
            kuno_image::validate_output(&png, recipe.width, recipe.height)?;
            Ok(png)
        };
        let output = tokio::select! {
            _ = response.closed() => Err("image_cancelled"),
            _ = shutdown.wait_for(|value| *value) => Err("image_cancelled"),
            result = tokio::time::timeout(Duration::from_secs(config.timeout_seconds), work) =>
                result.unwrap_or(Err("image_generation_timeout")),
        };
        if output.is_ok() {
            // A final frame alone is insufficient: read_result requires CLEAN.
            return Finished {
                output,
                settled: true,
                forced: false,
            };
        }
        let mut worker = state.take().unwrap();
        let (settled, forced, diagnostic) =
            stop(&mut worker, !terminal_error && !cleanup_ack).await;
        let output = if diagnostic == Some("image_vulkan_device_lost") {
            Err("image_vulkan_device_lost")
        } else {
            output
        };
        Finished {
            output,
            settled,
            forced,
        }
    }
}

struct Payload(Vec<u8>);
impl Payload {
    fn clear(&mut self) {
        kuno_image::worker_protocol::clear_bytes(&mut self.0);
    }
}
impl Drop for Payload {
    fn drop(&mut self) {
        self.clear();
    }
}

async fn spawn(
    identity: Identity,
    config: &ImageConfig,
    sycl: bool,
) -> Result<Worker, &'static str> {
    let mut command = Command::new(&identity.executable);
    if sycl {
        command.args([
            "image-worker-launch",
            config.runtime.as_deref().unwrap(),
            config.models.to_str().unwrap(),
            &config.pci_address,
        ]);
    } else {
        command.args([
            "image-worker-resident",
            config.models.to_str().unwrap(),
            &config.pci_address,
        ]);
    }
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", "/nonexistent")
        .env("KUNO_IMAGE_RESIDENT", "1")
        .env("KUNO_IMAGE_PREVIEWS", "1")
        .env("MESA_SHADER_CACHE_DISABLE", "true")
        .env("__GL_SHADER_DISK_CACHE", "0")
        .env("CUDA_CACHE_DISABLE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let parent = std::process::id();
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) != 0
                || libc::getppid() as u32 != parent
            {
                return Err(std::io::Error::other("image_parent_unavailable"));
            }
            Ok(())
        });
    }
    let mut child = command.spawn().map_err(|_| "image_worker_spawn_failed")?;
    let input = child.stdin.take().unwrap();
    let output = child.stdout.take().unwrap();
    let diagnostic = tokio::spawn(crate::image_runtime::diagnostics(
        child.stderr.take().unwrap(),
    ));
    let mut worker = Worker {
        identity,
        child,
        input: Some(input),
        output,
        diagnostic,
    };
    let ready = tokio::time::timeout(Duration::from_secs(30), frame(&mut worker.output)).await;
    if !matches!(ready, Ok(Ok((4, bytes))) if bytes.is_empty()) {
        // No content has been dispatched, but don't lose ownership of this PID.
        let (settled, forced, _) = stop(&mut worker, true).await;
        return Err(if settled && !forced {
            "image_worker_not_ready"
        } else {
            "image_worker_cleanup_uncertain"
        });
    }
    Ok(worker)
}

async fn stop(worker: &mut Worker, active: bool) -> (bool, bool, Option<&'static str>) {
    if !active {
        // Orderly idle shutdown runs the native context destructor; avoid signals.
        if let Some(input) = &mut worker.input {
            let _ = input.write_all(b"KIR2\x02\0\0\0\0").await;
        }
    } else if let Some(pid) = worker.child.id() {
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
    }
    // Unix ChildStdin::shutdown is a no-op: close the actual pipe. This also
    // wakes a worker cancelled during a partially written request or after CLEAN.
    drop(worker.input.take());
    let cleanup = async {
        match tokio::time::timeout(Duration::from_secs(90), worker.child.wait()).await {
            Ok(Ok(status)) => {
                use std::os::unix::process::ExitStatusExt;
                (true, status.signal().is_some())
            }
            _ => (
                tokio::time::timeout(Duration::from_secs(10), worker.child.kill())
                    .await
                    .is_ok_and(|result| result.is_ok()),
                true,
            ),
        }
    };
    // A native final/preview write may already be blocked when HTTP disconnects.
    // Drain into a fixed-size discard buffer so cancellation can reach cleanup.
    let drain = async {
        let mut discarded = [0u8; 8192];
        loop {
            match worker.output.read(&mut discarded).await {
                Ok(0) | Err(_) => break,
                Ok(_) => discarded.fill(0),
            }
        }
    };
    tokio::pin!(cleanup, drain);
    let (settled, forced) = tokio::select! {
        result = &mut cleanup => result,
        _ = &mut drain => cleanup.await,
    };
    let diagnostic = tokio::time::timeout(Duration::from_secs(1), &mut worker.diagnostic)
        .await
        .ok()
        .and_then(Result::ok)
        .flatten();
    (settled, forced, diagnostic)
}

async fn frame<R: tokio::io::AsyncRead + Unpin>(
    pipe: &mut R,
) -> Result<(u8, Vec<u8>), &'static str> {
    let mut header = [0; 9];
    pipe.read_exact(&mut header)
        .await
        .map_err(|_| "image_worker_output_failed")?;
    if &header[..4] != b"KIP1" || !(1..=5).contains(&header[4]) {
        return Err("image_worker_frame_invalid");
    }
    let size = u32::from_be_bytes(header[5..9].try_into().unwrap()) as usize;
    let maximum = match header[4] {
        1 => kuno_image::preview::MAX_PREVIEW_BYTES,
        2 => 20 * 1024 * 1024,
        5 => 128,
        _ => 0,
    };
    if size > maximum {
        return Err("image_worker_output_too_large");
    }
    let mut bytes = vec![0; size];
    pipe.read_exact(&mut bytes)
        .await
        .map_err(|_| "image_worker_output_failed")?;
    Ok((header[4], bytes))
}

async fn read_result<R: tokio::io::AsyncRead + Unpin>(
    pipe: &mut R,
    previews: &watch::Sender<Option<kuno_image::preview::Preview>>,
    steps: u32,
    terminal_error: &mut bool,
    cleanup_ack: &mut bool,
) -> Output {
    let mut result = None;
    let mut count = 0;
    loop {
        let (kind, bytes) = frame(pipe).await?;
        match kind {
            1 if result.is_none() => {
                count += 1;
                if count > steps.saturating_mul(2) + 2 {
                    return Err("image_worker_frame_invalid");
                }
                let preview = kuno_image::preview::Preview::from_frame(&bytes)?;
                if preview.steps != steps {
                    return Err("image_preview_invalid");
                }
                previews.send_replace(Some(preview));
            }
            2 if result.is_none() => result = Some(bytes),
            3 if result.is_some() && bytes.is_empty() => {
                *cleanup_ack = true;
                return Ok(result.unwrap());
            }
            5 => {
                *terminal_error = true;
                return Err(match bytes.as_slice() {
                    b"image_vulkan_device_lost" => "image_vulkan_device_lost",
                    b"image_cancelled" => "image_cancelled",
                    b"image_model_load_failed" => "image_model_load_failed",
                    b"image_gpu_memory_unavailable" => "image_gpu_memory_unavailable",
                    b"image_generation_failed" => "image_generation_failed",
                    b"image_attention_unsupported" => "image_attention_unsupported",
                    b"image_request_cleanup_failed" => "image_request_cleanup_failed",
                    _ => "image_worker_failed",
                });
            }
            _ => return Err("image_worker_frame_invalid"),
        }
    }
}
