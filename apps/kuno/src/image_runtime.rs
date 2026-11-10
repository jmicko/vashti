//! One request per short-lived native process. Inference content travels only
//! through pipes. The caller keeps its admission lease until cleanup finishes.
use kuno_protocol::ImageRecipe;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
    sync::oneshot,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageConfig {
    pub models: PathBuf,
    pub pci_address: String,
    pub timeout_seconds: u64,
    #[serde(default = "default_collections")]
    pub collections: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "crate::image_engine::ImageEngine::is_native"
    )]
    pub engine: crate::image_engine::ImageEngine,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
}
fn default_collections() -> Vec<String> {
    vec!["qwen-image-2512-fp8".into()]
}
impl ImageConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        use crate::image_engine::{ImageEngine, SYCL_RUNTIME};
        match self.engine {
            ImageEngine::Native if self.runtime.is_some() => {
                return Err("image_runtime_configuration_invalid");
            }
            ImageEngine::ManagedSycl if self.runtime.as_deref() != Some(SYCL_RUNTIME) => {
                return Err("image_runtime_configuration_invalid");
            }
            _ => {}
        }
        // Every known collection may be enabled at once; duplicates are rejected.
        if self.collections.is_empty()
            || self.collections.len() > crate::collections::KNOWN_COLLECTIONS
            || (1..self.collections.len())
                .any(|i| self.collections[..i].contains(&self.collections[i]))
            || self
                .collections
                .iter()
                .any(|id| crate::collections::collection(id).is_err())
        {
            return Err("image_collections_invalid");
        }
        let pci = self.pci_address.as_bytes();
        if !self.models.is_absolute()
            || self.models.to_str().is_none()
            || pci.len() != 12
            || !pci.iter().enumerate().all(|(i, c)| match i {
                4 | 7 => *c == b':',
                10 => *c == b'.',
                _ => c.is_ascii_hexdigit(),
            })
            || !(1..=3600).contains(&self.timeout_seconds)
        {
            return Err("image_configuration_invalid");
        }
        Ok(())
    }
    /// Whether a MiniMax video collection is enabled (it needs the managed SYCL engine).
    pub fn video_enabled(&self) -> bool {
        self.collections.iter().any(|id| {
            id == kuno_protocol::MINIMAX_H3_COLLECTION
                || id == kuno_protocol::MINIMAX_H3_REFERENCE_COLLECTION
        })
    }
    pub fn uses_sycl(&self, workflow: &str) -> bool {
        self.engine == crate::image_engine::ImageEngine::ManagedSycl
            && workflow != kuno_protocol::QWEN_RECIPE
    }
}
pub type Output = Result<Vec<u8>, &'static str>;
pub struct Finished {
    pub output: Output,
    /// False prevents admitting another model while a child may still exist.
    pub reaped: bool,
    /// A forced stop can leave the driver unhealthy even after process exit.
    pub forced: bool,
}

pub(crate) async fn diagnostics(mut stderr: tokio::process::ChildStderr) -> Option<&'static str> {
    // Raw backend text never reaches logs, errors, or persistent state. Keep
    // only a fixed error vocabulary emitted by our private worker entrypoint.
    const CODES: &[&str] = &[
        "image_physical_gpu_not_found",
        "image_reference_count_invalid",
        "image_reference_base64_invalid",
        "image_reference_png_invalid",
        "image_reference_format_unsupported",
        "image_reference_decode_failed",
        "image_reference_animation_unsupported",
        "image_reference_too_large",
        "image_reference_dimensions_invalid",
        "image_reference_aspect_ratio_unsupported",
        "image_model_load_failed",
        "image_generation_failed",
        "image_gpu_memory_unavailable",
        "image_native_runtime_failed",
        "image_vulkan_device_lost",
        "image_gpu_int8_unsupported",
        "image_attention_unsupported",
        "image_requires_landlock_abi_3",
        "image_filesystem_rules_failed",
        "image_device_rules_failed",
        "image_filesystem_restriction_failed",
        "image_requires_memory_swap_max_zero",
        "image_requires_cgroup_v2_no_swap",
        "image_swap_status_unavailable",
        "image_cgroup_unavailable",
        "image_managed_runtime_unavailable",
        "image_managed_runtime_invalid",
        "image_managed_runtime_incompatible",
        "image_managed_runtime_untrusted",
        "image_managed_runtime_corrupt",
        "image_managed_runtime_exec_failed",
    ];
    let mut buffer = [0; 1024];
    let mut line = Vec::with_capacity(128);
    let mut oversized = false;
    let mut result = None;
    while let Ok(n) = stderr.read(&mut buffer).await {
        if n == 0 {
            break;
        }
        for &byte in &buffer[..n] {
            if byte == b'\n' {
                if !oversized
                    && let Some(source) = line.strip_prefix(b"kuno-image: backend-error-source ")
                    && let Ok(source) = std::str::from_utf8(source)
                    && let Some((file, number)) = source.split_once(':')
                    && [
                        "image.cpp",
                        "conditioner.hpp",
                        "model_loader.cpp",
                        "model_manager.cpp",
                        "diffusion_engine.cpp",
                        "ggml_extend_backend.cpp",
                        "compute_workspace.cpp",
                        "model_builders.cpp",
                        "runner.cpp",
                        "ggml_extend.hpp",
                        "graph_executor.cpp",
                        "graph_partition.cpp",
                    ]
                    .contains(&file)
                    && let Ok(number) = number.parse::<u32>()
                    && (1..=50000).contains(&number)
                {
                    eprintln!("kuno: image_backend_failure {file}:{number}");
                }
                if !oversized
                    && result != Some("image_vulkan_device_lost")
                    && let Some(code) = line.strip_prefix(b"kuno: ")
                {
                    result = CODES
                        .iter()
                        .copied()
                        .find(|known| known.as_bytes() == code)
                        .or(result);
                }
                line.clear();
                oversized = false;
            } else if line.len() < 128 {
                line.push(byte);
            } else {
                oversized = true;
            }
        }
    }
    result
}
pub async fn run(
    executable: &Path,
    config: &ImageConfig,
    recipe: ImageRecipe,
    response: &mut oneshot::Sender<Output>,
    shutdown: Option<tokio::sync::watch::Receiver<bool>>,
) -> Finished {
    run_with_previews(executable, config, recipe, response, shutdown, None).await
}
#[allow(clippy::too_many_arguments)]
pub async fn run_with_previews(
    executable: &Path,
    config: &ImageConfig,
    recipe: ImageRecipe,
    response: &mut oneshot::Sender<Output>,
    shutdown: Option<tokio::sync::watch::Receiver<bool>>,
    previews: Option<tokio::sync::watch::Sender<Option<kuno_image::preview::Preview>>>,
) -> Finished {
    let fail = |code| Finished {
        output: Err(code),
        reaped: true,
        forced: false,
    };
    if let Err(code) = recipe.validate_execution().and_then(|_| config.validate()) {
        return fail(code);
    }
    if recipe.seed.parse::<i64>().is_err() {
        return fail("native_image_seed_range_unsupported");
    }
    if response.is_closed() {
        return fail("image_cancelled");
    }
    let mut command = Command::new(executable);
    if config.uses_sycl(&recipe.workflow) {
        let id = config.runtime.as_deref().unwrap();
        if let Err(code) = crate::image_engine::VerifiedRuntime::load(id) {
            return fail(code);
        }
        command.args([
            "image-worker-launch",
            id,
            config.models.to_str().unwrap(),
            &config.pci_address,
        ]);
    } else {
        command.args([
            "image-worker",
            config.models.to_str().unwrap(),
            &config.pci_address,
        ]);
    }
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", "/nonexistent")
        .env("MESA_SHADER_CACHE_DISABLE", "true")
        .env("__GL_SHADER_DISK_CACHE", "0")
        .env("CUDA_CACHE_DISABLE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if previews.is_some() {
        command.env("KUNO_IMAGE_PREVIEWS", "1");
    }
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
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(_) => return fail("image_worker_spawn_failed"),
    };
    let mut input = child.stdin.take().unwrap();
    let mut diagnostic = tokio::spawn(diagnostics(child.stderr.take().unwrap()));
    let mut output = child.stdout.take().unwrap();
    let work = async {
        let payload = serde_json::to_vec(&recipe).map_err(|_| "image_request_encoding_failed")?;
        input
            .write_all(&payload)
            .await
            .map_err(|_| "image_worker_input_failed")?;
        drop(input);
        let png = if let Some(previews) = previews {
            read_frames(&mut output, &previews, recipe.steps).await?
        } else {
            let mut png = Vec::new();
            output
                .take(20 * 1024 * 1024 + 1)
                .read_to_end(&mut png)
                .await
                .map_err(|_| "image_worker_output_failed")?;
            if png.len() > 20 * 1024 * 1024 {
                return Err("image_worker_output_too_large");
            }
            png
        };
        let status = child.wait().await.map_err(|_| "image_worker_wait_failed")?;
        if !status.success() {
            return Err("image_worker_failed");
        }
        kuno_image::validate_output(&png, recipe.width, recipe.height)?;
        Ok(png)
    };
    let shutdown = async {
        if let Some(mut receiver) = shutdown {
            let _ = receiver.wait_for(|stopped| *stopped).await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    let mut output = tokio::select! {
        _ = response.closed()=>Err("image_cancelled"),
        _ = shutdown=>Err("image_cancelled"),
        result = tokio::time::timeout(Duration::from_secs(config.timeout_seconds),work)=>
            result.unwrap_or(Err("image_generation_timeout")),
    };
    let forced;
    let reaped = match child.try_wait() {
        Ok(Some(status)) => {
            use std::os::unix::process::ExitStatusExt;
            forced = status.signal().is_some();
            true
        }
        _ => {
            // Let the engine finish a GPU submission before destroying its
            // context. Forced termination can leave some drivers unhealthy.
            if let Some(pid) = child.id() {
                unsafe {
                    libc::kill(pid as i32, libc::SIGTERM);
                }
            }
            if let Ok(Ok(status)) =
                tokio::time::timeout(Duration::from_secs(90), child.wait()).await
            {
                use std::os::unix::process::ExitStatusExt;
                forced = status.signal().is_some();
                true
            } else {
                forced = true;
                tokio::time::timeout(Duration::from_secs(10), child.kill())
                    .await
                    .is_ok_and(|r| r.is_ok())
            }
        }
    };
    if reaped
        && let Ok(Ok(Some(code))) =
            tokio::time::timeout(Duration::from_secs(1), &mut diagnostic).await
        && (code == "image_vulkan_device_lost"
            || matches!(
                output,
                Err("image_worker_failed" | "image_worker_input_failed")
            ))
    {
        output = Err(code);
    }
    diagnostic.abort();
    Finished {
        output,
        reaped,
        forced,
    }
}

async fn read_frames<R: tokio::io::AsyncRead + Unpin>(
    pipe: &mut R,
    sender: &tokio::sync::watch::Sender<Option<kuno_image::preview::Preview>>,
    steps: u32,
) -> Output {
    let mut count = 0;
    loop {
        let mut header = [0u8; 9];
        pipe.read_exact(&mut header)
            .await
            .map_err(|_| "image_worker_output_failed")?;
        if &header[..4] != b"KIP1" || !matches!(header[4], 1 | 2) {
            return Err("image_worker_frame_invalid");
        }
        let size = u32::from_be_bytes(header[5..9].try_into().unwrap()) as usize;
        let limit = if header[4] == 1 {
            kuno_image::preview::MAX_PREVIEW_BYTES
        } else {
            20 * 1024 * 1024
        };
        if size > limit {
            return Err("image_worker_output_too_large");
        }
        let mut bytes = vec![0; size];
        pipe.read_exact(&mut bytes)
            .await
            .map_err(|_| "image_worker_output_failed")?;
        if header[4] == 2 {
            let mut extra = [0];
            if pipe
                .read(&mut extra)
                .await
                .map_err(|_| "image_worker_output_failed")?
                != 0
            {
                return Err("image_worker_frame_invalid");
            }
            return Ok(bytes);
        }
        count += 1;
        if count > steps.saturating_mul(2) + 2 {
            return Err("image_worker_frame_invalid");
        }
        let preview = kuno_image::preview::Preview::from_frame(&bytes)?;
        if preview.steps != steps {
            return Err("image_preview_invalid");
        }
        // Only the newest frame is retained. HTTP readers cannot block sampling.
        sender.send_replace(Some(preview));
    }
}

#[cfg(test)]
mod preview_tests {
    use super::*;
    fn frame(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut frame = b"KIP1".to_vec();
        frame.push(kind);
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }
    #[tokio::test]
    async fn preview_transport_keeps_latest_without_a_subscriber() {
        let png = include_bytes!("../../vashti/src/images/fixtures/64.png");
        let mut pipe = Vec::new();
        for step in [1u32, 2] {
            let mut payload = Vec::new();
            for number in [step, 2, 64, 64] {
                payload.extend_from_slice(&number.to_be_bytes());
            }
            payload.extend_from_slice(png);
            pipe.extend(frame(1, &payload));
        }
        pipe.extend(frame(2, png));
        let sender = tokio::sync::watch::channel(None).0;
        let result = read_frames(&mut pipe.as_slice(), &sender, 2).await.unwrap();
        assert_eq!(result, png);
        assert_eq!(sender.borrow().as_ref().unwrap().step, 2);
        let mut invalid = frame(2, png);
        invalid.push(0);
        assert!(
            read_frames(&mut invalid.as_slice(), &sender, 2)
                .await
                .is_err()
        );
        assert!(
            read_frames(&mut b"bogus".as_slice(), &sender, 2)
                .await
                .is_err()
        );
        let oversized = [b"KIP1".as_slice(), &[1], &(2_000_000u32.to_be_bytes())].concat();
        assert!(
            read_frames(&mut oversized.as_slice(), &sender, 2)
                .await
                .is_err()
        );
    }
}
