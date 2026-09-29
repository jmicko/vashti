//! Owned Ollama workers. No inherited inference environment, content logs, or
//! adoption of an arbitrary listener. Startup verifies the selected physical GPU.
use crate::lifecycle::{Outcome, Span, Stage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    net::SocketAddr,
    path::PathBuf,
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    io::AsyncReadExt,
    process::{Child, Command},
};

type Result<T> = std::result::Result<T, &'static str>;

/// Bounds how long a stopped worker may delay its own scope release while a
/// killed runner is still visible to the kernel.
const GROUP_EXIT_PROBES: u32 = 60;
const GROUP_EXIT_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeviceSelection {
    Cuda { uuid: String, pci_address: String },
    Vulkan { index: u32, pci_address: String },
}

impl DeviceSelection {
    pub fn pci_address(&self) -> &str {
        match self {
            Self::Cuda { pci_address, .. } | Self::Vulkan { pci_address, .. } => pci_address,
        }
    }
    fn matches(&self, device: &DiscoveredDevice) -> bool {
        let pci = if device.pci.matches(':').count() == 1 {
            format!("0000:{}", device.pci)
        } else {
            device.pci.clone()
        };
        if !self.pci_address().eq_ignore_ascii_case(&pci) {
            return false;
        }
        match self {
            Self::Cuda { uuid, .. } => {
                device.library.eq_ignore_ascii_case("cuda") && &device.filter_id == uuid
            }
            Self::Vulkan { .. } => device.library.eq_ignore_ascii_case("vulkan"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerConfig {
    pub executable: PathBuf,
    pub models: PathBuf,
    pub home: PathBuf,
    pub address: SocketAddr,
    pub device: DeviceSelection,
    pub max_loaded_models: u32,
    pub context_length: u32,
}

impl WorkerConfig {
    pub fn validate(&self) -> Result<()> {
        let pci = self.device.pci_address();
        let valid_pci = pci.len() == 12
            && pci.bytes().enumerate().all(|(i, c)| match i {
                4 | 7 => c == b':',
                10 => c == b'.',
                _ => c.is_ascii_hexdigit(),
            });
        if !valid_pci
            || !self.address.ip().is_loopback()
            || self.address.port() == 0
            || !self.executable.is_absolute()
            || !self.models.is_absolute()
            || !self.home.is_absolute()
            || [&self.executable, &self.models, &self.home]
                .iter()
                .any(|p| p.to_str().is_none())
            || !(1..=64).contains(&self.max_loaded_models)
            || !(512..=1_048_576).contains(&self.context_length)
        {
            return Err("worker_configuration_invalid");
        }
        match &self.device {
            DeviceSelection::Cuda { uuid, .. }
                if !uuid.starts_with("GPU-")
                    || uuid.len() != 40
                    || !uuid[4..]
                        .bytes()
                        .all(|c| c.is_ascii_hexdigit() || c == b'-') =>
            {
                return Err("worker_gpu_uuid_invalid");
            }
            DeviceSelection::Vulkan { index, .. } if *index > 63 => {
                return Err("worker_gpu_index_invalid");
            }
            _ => (),
        }
        Ok(())
    }

    fn environment(&self) -> BTreeMap<String, String> {
        let mut env: BTreeMap<String, String> = [
            ("PATH", "/usr/local/bin:/usr/bin:/bin"),
            ("HOME", self.home.to_str().unwrap_or("")),
            ("OLLAMA_MODELS", self.models.to_str().unwrap_or("")),
            ("OLLAMA_NOPRUNE", "true"),
            ("OLLAMA_NO_CLOUD", "true"),
            ("OLLAMA_DEBUG", "false"),
            ("OLLAMA_DEBUG_LOG_REQUESTS", "false"),
            ("OLLAMA_NUM_PARALLEL", "1"),
            ("OLLAMA_KEEP_ALIVE", "-1"),
            ("CUDA_VISIBLE_DEVICES", "-1"),
            ("ROCR_VISIBLE_DEVICES", "-1"),
            ("HIP_VISIBLE_DEVICES", "-1"),
            ("GGML_VK_VISIBLE_DEVICES", "-1"),
            ("OLLAMA_VULKAN", "false"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
        env.insert("OLLAMA_HOST".into(), self.address.to_string());
        env.insert(
            "OLLAMA_MAX_LOADED_MODELS".into(),
            self.max_loaded_models.to_string(),
        );
        env.insert(
            "OLLAMA_CONTEXT_LENGTH".into(),
            self.context_length.to_string(),
        );
        match &self.device {
            DeviceSelection::Cuda { uuid, .. } => {
                env.insert("CUDA_VISIBLE_DEVICES".into(), uuid.clone());
            }
            DeviceSelection::Vulkan { index, .. } => {
                env.insert("OLLAMA_VULKAN".into(), "true".into());
                env.insert("GGML_VK_VISIBLE_DEVICES".into(), index.to_string());
            }
        }
        env
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DiscoveredDevice {
    library: String,
    id: String,
    filter_id: String,
    pci: String,
}

#[derive(Default)]
struct Diagnostics {
    devices: Vec<DiscoveredDevice>,
    invalid: bool,
}

impl Diagnostics {
    fn observe(&mut self, line: &[u8]) {
        // Only startup hardware identity survives this function. Request/error
        // text is discarded. No unbounded lines, raw rings, or stderr files.
        let Ok(text) = std::str::from_utf8(line) else {
            return;
        };
        if !text.contains("msg=\"inference compute\"") {
            return;
        }
        let fields = log_fields(text);
        let Some(fields) = fields else {
            self.invalid = true;
            return;
        };
        let Some(library) = fields.get("library") else {
            self.invalid = true;
            return;
        };
        let Some(id) = fields.get("id") else {
            self.invalid = true;
            return;
        };
        let Some(pci) = fields.get("pci_id") else {
            self.invalid = true;
            return;
        };
        let filter_id = fields.get("filter_id").cloned().unwrap_or_default();
        if library.len() > 16
            || id.len() > 128
            || filter_id.len() > 128
            || pci.len() > 16
            || self.devices.len() >= 64
        {
            self.invalid = true;
            return;
        }
        let d = DiscoveredDevice {
            library: library.clone(),
            id: id.clone(),
            filter_id,
            pci: pci.clone(),
        };
        if !self.devices.contains(&d) {
            self.devices.push(d);
        }
    }
    fn verified(&self, selection: &DeviceSelection) -> bool {
        !self.invalid && self.devices.len() == 1 && selection.matches(&self.devices[0])
    }
}

fn log_fields(text: &str) -> Option<BTreeMap<String, String>> {
    let mut rest = text.trim();
    let mut result = BTreeMap::new();
    while !rest.is_empty() {
        let (key, value) = rest.split_once('=')?;
        if key.is_empty() || key.bytes().any(|b| b.is_ascii_whitespace()) {
            return None;
        }
        let (value, remaining) = if let Some(quoted) = value.strip_prefix('"') {
            let end = quoted.find('"')?;
            // We only need ASCII hardware fields. Escaped or unfamiliar formats
            // cannot establish device identity.
            if quoted[..end].contains('\\') {
                return None;
            }
            (&quoted[..end], &quoted[end + 1..])
        } else {
            let end = value.find(char::is_whitespace).unwrap_or(value.len());
            (&value[..end], &value[end..])
        };
        if matches!(key, "library" | "id" | "filter_id" | "pci_id")
            && result.insert(key.into(), value.into()).is_some()
        {
            return None;
        }
        rest = remaining.trim_start();
    }
    Some(result)
}

pub struct Worker {
    config: WorkerConfig,
    child: Child,
    diagnostics: Arc<Mutex<Diagnostics>>,
    drain: tokio::task::JoinHandle<()>,
    client: reqwest::Client,
}

impl Worker {
    pub async fn start(config: WorkerConfig) -> Result<Self> {
        let mut worker = Self::spawn(config)?;
        worker.ready().await?;
        Ok(worker)
    }

    // No await between process creation and returning its ownership. Supervisors
    // retain this handle before waiting for discovery, including cancellation.
    pub(crate) fn spawn(config: WorkerConfig) -> Result<Self> {
        config.validate()?;
        let observed = Span::new(Stage::WorkerSpawn, None, Some(config.device.pci_address()));
        let result = (|| {
            if !config.models.is_dir() {
                return Err("worker_models_directory_unavailable");
            }
            if !config.home.is_dir() {
                return Err("worker_home_directory_unavailable");
            }
            if !config.executable.is_file() {
                return Err("worker_executable_unavailable");
            }
            let client = reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .pool_max_idle_per_host(0)
                .timeout(Duration::from_secs(300))
                .build()
                .map_err(|_| "worker_client_failed")?;
            // Do not disturb another listener. Ownership is checked again after
            // spawn, since this preflight alone cannot exclude a port race.
            let probe = std::net::TcpListener::bind(config.address)
                .map_err(|_| "worker_port_unavailable")?;
            let mut command = Command::new(&config.executable);
            command
                .arg("serve")
                .env_clear()
                .envs(config.environment())
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            command.process_group(0);
            let parent_pid = std::process::id();
            // SAFETY: only async-signal-safe syscalls in the forked child. The
            // systemd service's cgroup additionally owns all runner descendants.
            unsafe {
                command.pre_exec(move || {
                    let limit = libc::rlimit {
                        rlim_cur: 0,
                        rlim_max: 0,
                    };
                    if libc::setrlimit(libc::RLIMIT_CORE, &limit) != 0
                        || libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::getppid() != parent_pid as i32 {
                        libc::_exit(1);
                    }
                    Ok(())
                });
            }
            drop(probe);
            let mut child = command.spawn().map_err(|_| "worker_spawn_failed")?;
            let mut stderr = child
                .stderr
                .take()
                .ok_or("worker_diagnostics_unavailable")?;
            let diagnostics = Arc::new(Mutex::new(Diagnostics::default()));
            let sink = diagnostics.clone();
            let drain = tokio::spawn(async move {
                let mut bytes = [0; 4096];
                let mut line = Vec::with_capacity(4096);
                let mut overflow = false;
                while let Ok(n) = stderr.read(&mut bytes).await {
                    if n == 0 {
                        break;
                    }
                    for &byte in &bytes[..n] {
                        if byte == b'\n' {
                            if !overflow {
                                sink.lock().unwrap().observe(&line);
                            }
                            line.clear();
                            overflow = false;
                        } else if line.len() < 16384 && !overflow {
                            line.push(byte);
                        } else {
                            overflow = true;
                            line.clear();
                        }
                    }
                }
            });
            Ok(Self {
                config,
                child,
                diagnostics,
                drain,
                client,
            })
        })();
        observed.result(&result);
        result
    }

    pub(crate) async fn ready(&mut self) -> Result<()> {
        let observed = Span::new(
            Stage::WorkerReady,
            self.child.id(),
            Some(self.config.device.pci_address()),
        );
        let ready = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if self.exited()? {
                    return Err("worker_exited_during_startup");
                }
                let owns = crate::systemd::owns_listener(
                    self.child.id().ok_or("worker_pid_unavailable")?,
                    self.config.address,
                )
                .is_ok();
                let verified = self
                    .diagnostics
                    .lock()
                    .unwrap()
                    .verified(&self.config.device);
                if owns && verified && self.metadata("/api/version").await.is_ok() {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await;
        let ready = match ready {
            Ok(result) => {
                observed.result(&result);
                result
            }
            Err(_) => {
                observed.finish(Outcome::Timeout);
                Err("worker_startup_or_device_verification_failed")
            }
        };
        if let Err(mut e) = ready {
            if e == "worker_startup_or_device_verification_failed" {
                let d = self.diagnostics.lock().unwrap();
                e = if d.devices.is_empty() {
                    "worker_gpu_discovery_unavailable"
                } else if !d.verified(&self.config.device) {
                    "worker_selected_gpu_mismatch"
                } else {
                    "worker_listener_or_health_verification_failed"
                };
            }
            self.stop().await?;
            return Err(e);
        }
        Ok(())
    }

    pub fn address(&self) -> SocketAddr {
        self.config.address
    }
    pub fn process_id(&self) -> Option<u32> {
        self.child.id()
    }
    pub fn device(&self) -> &DeviceSelection {
        &self.config.device
    }

    fn exited(&self) -> Result<bool> {
        let pid = self.child.id().ok_or("worker_pid_unavailable")?;
        // Keep an exited leader unreaped until stop() has signalled its group.
        // try_wait() would release its PID while runners may still be alive.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        // SAFETY: valid output pointer and a child PID owned by this Worker;
        // WNOWAIT observes exit without releasing the process-group identity.
        let result = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if result != 0 {
            return Err("worker_status_failed");
        }
        // SAFETY: waitid initialized the siginfo child-status variant.
        Ok(unsafe { info.si_pid() } != 0)
    }

    pub fn verify_alive(&mut self) -> Result<()> {
        if self.exited()?
            || !self
                .diagnostics
                .lock()
                .unwrap()
                .verified(&self.config.device)
        {
            return Err("worker_unavailable_or_device_changed");
        }
        crate::systemd::owns_listener(
            self.child.id().ok_or("worker_pid_unavailable")?,
            self.config.address,
        )
    }

    async fn metadata(&self, path: &str) -> Result<Value> {
        let response = self
            .client
            .get(format!("http://{}{path}", self.config.address))
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map_err(|_| "worker_metadata_unavailable")?;
        bounded_json(response).await
    }

    pub async fn running(&mut self) -> Result<crate::resources::Residency> {
        self.verify_alive()?;
        let value = self.metadata("/api/ps").await?;
        let bytes = serde_json::to_vec(&value).map_err(|_| "worker_metadata_invalid")?;
        let mut residency = crate::resources::parse_residency(&bytes)?;
        for model in &mut residency.models {
            model.device_assignment_verified = true;
        }
        Ok(residency)
    }

    /// Caller holds the worker/device reservation. No user content is sent.
    /// Any spill is unloaded before a caller can dispatch useful inference.
    pub async fn load(&mut self, model: &str, digest: &str, context: u32) -> Result<()> {
        let result = self.load_supervised(model, digest, context).await;
        if result.is_err() {
            self.stop().await?;
        }
        result
    }

    /// The managed supervisor retains ownership and captures memory evidence
    /// BEFORE cleanup. It must stop/reap on error or cancellation, and must not
    /// dispatch user input until this succeeds.
    pub(crate) async fn load_supervised(
        &mut self,
        model: &str,
        digest: &str,
        context: u32,
    ) -> Result<()> {
        let observed = Span::new(
            Stage::WorkerLoad,
            self.child.id(),
            Some(self.config.device.pci_address()),
        );
        let result = async {
            self.verify_alive()?;
            if model.is_empty()
                || model.len() > 512
                || model.chars().any(char::is_control)
                || digest.len() != 64
                || !digest.bytes().all(|c| c.is_ascii_hexdigit())
                || !(512..=1_048_576).contains(&context)
            {
                return Err("worker_load_identity_invalid");
            }
            let headers = Span::new(
                Stage::WorkerLoadHeaders,
                self.child.id(),
                Some(self.config.device.pci_address()),
            );
            let response = self.client.post(format!("http://{}/api/embeddings", self.config.address))
                    .json(&json!({"model":model,"prompt":"","stream":false,"keep_alive":-1,"options":{"num_ctx":context}}))
                    .send().await;
            let response = match response {
                Ok(response) => {
                    headers.finish(Outcome::Success);
                    response
                }
                Err(error) => {
                    headers.finish(transport_outcome(&error));
                    return Err("worker_load_failed");
                }
            };
            let value = bounded_json_observed(
                response,
                Some(Span::new(
                    Stage::WorkerLoadBody,
                    self.child.id(),
                    Some(self.config.device.pci_address()),
                )),
            ).await?;
            // Empty legacy embedding requests load without imposing either
            // completion or embedding capability on the selected model.
            if value
                .get("embedding")
                .and_then(Value::as_array)
                .is_none_or(|a| !a.is_empty())
            {
                return Err("worker_load_incomplete");
            }
            let state = self.running().await?;
            let loaded = state
                .models
                .iter()
                .find(|m| m.digest.as_deref() == Some(digest))
                .ok_or("worker_loaded_identity_mismatch")?;
            if loaded.residency != crate::resources::ReportedResidency::GpuReported {
                return Err("worker_cpu_spill_detected");
            }
            if loaded.context_length != Some(u64::from(context)) {
                return Err("worker_context_mismatch");
            }
            Ok(())
        }.await;
        observed.result(&result);
        result
    }

    pub async fn unload(&mut self, model: &str) -> Result<()> {
        let observed = Span::new(
            Stage::WorkerUnload,
            self.child.id(),
            Some(self.config.device.pci_address()),
        );
        let result = async {
            self.verify_alive()?;
            let headers = Span::new(
                Stage::WorkerUnloadHeaders,
                self.child.id(),
                Some(self.config.device.pci_address()),
            );
            let response = self
                .client
                .post(format!("http://{}/api/generate", self.config.address))
                .json(&json!({"model":model,"keep_alive":0,"stream":false}))
                .send()
                .await;
            let response = match response {
                Ok(response) => {
                    headers.finish(Outcome::Success);
                    response
                }
                Err(error) => {
                    headers.finish(transport_outcome(&error));
                    return Err("worker_unload_failed");
                }
            };
            let value = bounded_json_observed(
                response,
                Some(Span::new(
                    Stage::WorkerResponseBody,
                    self.child.id(),
                    Some(self.config.device.pci_address()),
                )),
            )
            .await?;
            if value.get("done").and_then(Value::as_bool) != Some(true) {
                return Err("worker_unload_incomplete");
            }
            Ok(()) // Scheduler must still observe actual VRAM release.
        }
        .await;
        observed.result(&result);
        result
    }

    /// Stops the owned process group and proves it is empty before the leader
    /// PID is released. Every caller either completes the worker/device
    /// reservation or lets it quarantine, so an unproven group must fail:
    /// a stale runner can still hold the device this scope reserved.
    pub async fn stop(&mut self) -> Result<()> {
        let observed = Span::new(
            Stage::WorkerStop,
            self.child.id(),
            Some(self.config.device.pci_address()),
        );
        let Some(pid) = self.child.id() else {
            // The leader is reaped only after its group was proven empty, so a
            // missing handle is itself the proof.
            self.drain.abort();
            observed.finish(Outcome::AlreadyStopped);
            return Ok(());
        };
        let result = async {
            Span::event(
                Stage::WorkerSignal,
                Outcome::Sigkill,
                Some(pid),
                Some(self.config.device.pci_address()),
            );
            // SAFETY: the child created its own process group and this handle is
            // still unreaped, so the PID cannot have been recycled.
            let signalled = unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
            if signalled != 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
            {
                Span::event(
                    Stage::WorkerSignal,
                    Outcome::Failed,
                    Some(pid),
                    Some(self.config.device.pci_address()),
                );
                return Err("worker_group_signal_failed");
            }
            let group = Span::new(
                Stage::WorkerGroupExit,
                Some(pid),
                Some(self.config.device.pci_address()),
            );
            if !wait_for_group_exit(pid).await {
                group.finish(Outcome::Timeout);
                // The leader stays unreaped so a later attempt can re-prove this
                // same group. Clearing the reservation now would be a guess.
                return Err("worker_group_not_empty");
            }
            group.finish(Outcome::Success);
            let reap = Span::new(
                Stage::WorkerReap,
                Some(pid),
                Some(self.config.device.pci_address()),
            );
            let reaped = self.child.wait().await.map_err(|_| "worker_reap_failed");
            reap.result(&reaped);
            reaped?;
            self.drain.abort();
            Ok(())
        }
        .await;
        observed.result(&result);
        result
    }
}

/// A killed process group can stay visible while a runner drains out of an
/// uninterruptible driver call. Zombies hold no device, so they count as gone.
async fn wait_for_group_exit(pgid: u32) -> bool {
    for _ in 0..GROUP_EXIT_PROBES {
        if process_group_empty(pgid) {
            return true;
        }
        tokio::time::sleep(GROUP_EXIT_INTERVAL).await;
    }
    false
}

fn process_group_empty(pgid: u32) -> bool {
    let Ok(entries) = fs::read_dir("/proc") else {
        return false;
    };
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        let stat = match fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => stat,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => return false,
        };
        let Some((_, rest)) = stat.rsplit_once(')') else {
            return false;
        };
        let mut fields = rest.split_whitespace();
        let Some(state) = fields.next() else {
            return false;
        };
        let Some(_ppid) = fields.next() else {
            return false;
        };
        let Some(group) = fields.next() else {
            return false;
        };
        if state != "Z" && group.parse::<u32>().is_ok_and(|group| group == pgid) {
            return false;
        }
    }
    true
}

impl Drop for Worker {
    fn drop(&mut self) {
        if let Some(pid) = self.child.id() {
            Span::event(
                Stage::WorkerDropSignal,
                Outcome::Sigkill,
                Some(pid),
                Some(self.config.device.pci_address()),
            );
            // SAFETY: same owned process-group boundary as stop().
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
        self.drain.abort();
    }
}

fn transport_outcome(error: &reqwest::Error) -> Outcome {
    if error.is_timeout() {
        Outcome::Timeout
    } else {
        Outcome::TransportError
    }
}

async fn bounded_json(response: reqwest::Response) -> Result<Value> {
    bounded_json_observed(response, None).await
}

async fn bounded_json_observed(
    mut response: reqwest::Response,
    observed: Option<Span>,
) -> Result<Value> {
    let mut outcome = Outcome::Success;
    let result = async {
        if !response.status().is_success() {
            outcome = Outcome::Rejected;
            return Err("worker_request_rejected");
        }
        let mut bytes = Vec::new();
        loop {
            let chunk = match response.chunk().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(error) => {
                    outcome = if error.is_timeout() {
                        Outcome::Timeout
                    } else {
                        Outcome::BodyError
                    };
                    return Err("worker_response_failed");
                }
            };
            if bytes.len() + chunk.len() > 2 * 1024 * 1024 {
                outcome = Outcome::InvalidResponse;
                return Err("worker_response_too_large");
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).map_err(|_| {
            outcome = Outcome::InvalidResponse;
            "worker_response_invalid"
        })
    }
    .await;
    if let Some(observed) = observed {
        observed.finish(outcome);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn observed_body_preserves_validation_errors_without_echoing_content() {
        for (status, body, expected) in [
            (503, "PRIVATE-UPSTREAM-ERROR", "worker_request_rejected"),
            (200, "PRIVATE-INVALID-JSON", "worker_response_invalid"),
        ] {
            let response = axum::http::Response::builder()
                .status(status)
                .body(body)
                .unwrap();
            let result = bounded_json_observed(
                reqwest::Response::from(response),
                Some(Span::new(
                    Stage::WorkerLoadBody,
                    Some(42),
                    Some("0000:2d:00.0"),
                )),
            )
            .await;
            assert_eq!(result, Err(expected));
        }
    }

    #[tokio::test]
    async fn preload_body_timeout_remains_a_response_failure() {
        use tokio::io::AsyncWriteExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 1024];
            assert!(socket.read(&mut request).await.unwrap() > 0);
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 32\r\n\r\n{")
                .await
                .unwrap();
            std::future::pending::<()>().await;
        });
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(100))
            .build()
            .unwrap();
        let response = client
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap();
        let result = bounded_json_observed(
            response,
            Some(Span::new(
                Stage::WorkerLoadBody,
                Some(42),
                Some("0000:2d:00.0"),
            )),
        )
        .await;
        server.abort();
        assert_eq!(result, Err("worker_response_failed"));
    }

    fn config() -> WorkerConfig {
        WorkerConfig {
            executable: "/usr/local/bin/ollama".into(),
            models: "/models".into(),
            home: "/worker-home".into(),
            address: "127.0.0.1:19001".parse().unwrap(),
            device: DeviceSelection::Cuda {
                uuid: "GPU-59ee3c71-80be-a06f-8d84-4a2d172d589c".into(),
                pci_address: "0000:04:00.0".into(),
            },
            max_loaded_models: 8,
            context_length: 4096,
        }
    }

    #[test]
    fn environment_controls_devices_capacity_and_content_logging() {
        let mut c = config();
        c.validate().unwrap();
        let env = c.environment();
        assert_eq!(env["OLLAMA_MAX_LOADED_MODELS"], "8");
        assert_eq!(env["OLLAMA_NUM_PARALLEL"], "1");
        assert_eq!(env["OLLAMA_DEBUG_LOG_REQUESTS"], "false");
        assert_eq!(env["GGML_VK_VISIBLE_DEVICES"], "-1");
        assert_eq!(env["ROCR_VISIBLE_DEVICES"], "-1");
        c.device = DeviceSelection::Vulkan {
            index: 1,
            pci_address: "0000:2d:00.0".into(),
        };
        let env = c.environment();
        assert_eq!(env["CUDA_VISIBLE_DEVICES"], "-1");
        assert_eq!(env["GGML_VK_VISIBLE_DEVICES"], "1");
        assert_eq!(env["OLLAMA_VULKAN"], "true");
        c.address = "0.0.0.0:19001".parse().unwrap();
        assert_eq!(c.validate(), Err("worker_configuration_invalid"));
    }

    #[test]
    fn discovery_requires_exact_physical_gpu_and_rejects_extra_devices() {
        let c = config();
        let mut d = Diagnostics::default();
        d.observe(b"time=2026 level=INFO msg=\"inference compute\" id=0 filter_id=GPU-59ee3c71-80be-a06f-8d84-4a2d172d589c library=CUDA name=\"RTX 3060\" pci_id=04:00.0 total=\"12.0 GiB\"");
        assert!(d.verified(&c.device));
        assert!(!d.verified(&DeviceSelection::Cuda {
            uuid: "GPU-wrong".into(),
            pci_address: "0000:04:00.0".into()
        }));
        d.observe(b"msg=\"inference compute\" library=Vulkan id=Vulkan0 pci_id=2d:00.0");
        assert!(!d.verified(&c.device));
    }

    #[test]
    fn request_text_is_discarded_and_ambiguous_discovery_is_rejected() {
        let mut d = Diagnostics::default();
        d.observe(b"request prompt=super-secret-message");
        assert!(d.devices.is_empty());
        d.observe(b"msg=\"inference compute\" library=CUDA id=a id=b pci_id=04:00.0");
        assert!(d.invalid);
        assert!(!d.verified(&config().device));
    }

    #[test]
    fn process_group_probe_fails_closed_for_a_live_group() {
        let pgid = unsafe { libc::getpgrp() as u32 };
        assert!(!process_group_empty(pgid));
    }

    #[tokio::test]
    async fn a_killed_owned_group_is_proven_empty_before_it_is_released() {
        use std::os::unix::process::CommandExt;
        let mut command = std::process::Command::new("/bin/sleep");
        command.arg("30").process_group(0);
        let mut child = command.spawn().expect("sleep");
        let pgid = child.id();
        assert!(!process_group_empty(pgid));
        // SAFETY: the child created its own process group and is still running.
        assert_eq!(unsafe { libc::kill(-(pgid as i32), libc::SIGKILL) }, 0);
        assert!(wait_for_group_exit(pgid).await);
        assert!(process_group_empty(u32::MAX));
        child.wait().unwrap();
    }
}
