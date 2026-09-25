//! Placement plans executed against a Kuno-owned Ollama worker. The returned
//! job owns its reservation until verified upstream completion. This supervisor
//! is separate from the legacy opaque gateway until routing is fully integrated.
use crate::{
    learning::{Failure, FailureKind, History, MemorySample, Outcome, ResidentWorkload},
    placement::{self, Candidate, Decision, Device, Requirement, Resident, Snapshot, Workload},
    reservations::{Reservation, Reservations, Resource, Scope},
    worker::{Worker, WorkerConfig},
};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, OwnedMutexGuard};

type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone)]
pub struct LoadRequest {
    pub model: String,
    pub digest: String,
    pub context: u32,
    /// Estimate for this exact workload on this backend, including KV/cache
    /// and workspace. Unknown workloads need an isolated profiling operation;
    /// a file-size estimate must not be labelled a measured profile.
    pub resident_bytes: u64,
    pub peak_bytes: u64,
}

struct Record {
    workload: Workload,
    context: u32,
    last_used: u64,
}
struct State {
    worker: Option<Worker>,
    records: BTreeMap<String, Record>,
    sequence: u64,
    profiles: BTreeMap<(String, u32), (u64, u64)>,
}

pub struct ManagedWorker {
    config: WorkerConfig,
    id: String,
    reserve: u64,
    reservations: Arc<Reservations>,
    state: Arc<Mutex<State>>,
    pid: AtomicU32,
    // Separate from the job-held state lock so diagnostics stay responsive.
    history: std::sync::Mutex<History>,
}

impl ManagedWorker {
    pub fn new(
        config: WorkerConfig,
        id: String,
        reserve: u64,
        reservations: Arc<Reservations>,
    ) -> Result<Arc<Self>> {
        config.validate()?;
        if id.is_empty() || id.len() > 128 || id.chars().any(char::is_control) || reserve == 0 {
            return Err("managed_worker_configuration_invalid");
        }
        Ok(Arc::new(Self {
            config,
            id,
            reserve,
            reservations,
            state: Arc::new(Mutex::new(State {
                worker: None,
                records: BTreeMap::new(),
                sequence: 0,
                profiles: BTreeMap::new(),
            })),
            pid: AtomicU32::new(0),
            history: std::sync::Mutex::new(History::default()),
        }))
    }

    fn device_id(&self) -> String {
        format!("pci:{}", self.config.device.pci_address())
    }
    fn scope(&self) -> Scope {
        [
            Resource::Device(self.device_id()),
            Resource::Worker(self.id.clone()),
        ]
        .into()
    }

    /// Stops a quarantined owned worker before reopening its reservation. Does
    /// not affect other workers or infer that failed combinations are unsafe.
    pub async fn recover(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        if !self
            .reservations
            .quarantined()
            .contains(&Resource::Worker(self.id.clone()))
        {
            return Err("managed_worker_not_quarantined");
        }
        if let Some(worker) = &mut state.worker {
            worker.stop().await?;
        }
        state.worker = None;
        self.pid.store(0, Ordering::Release);
        state.records.clear();
        self.reservations.reconciled(&self.scope())
    }

    async fn spawn_locked(&self, state: &mut State) -> Result<()> {
        if state.worker.is_none() {
            let samples = self.unloaded_samples().await;
            self.history.lock().unwrap().calibrate(&samples);
            let worker = Worker::spawn(self.config.clone())?;
            self.pid.store(
                worker.process_id().ok_or("worker_pid_unavailable")?,
                Ordering::Release,
            );
            state.worker = Some(worker);
            state.records.clear();
            state.profiles.clear();
        }
        Ok(())
    }

    /// Establish device ownership before an installed gateway accepts traffic.
    /// This performs discovery only; no model or user content is loaded.
    pub async fn initialize(&self) -> Result<()> {
        let mut lease = self.reservations.acquire(self.scope()).await?;
        let mut state = self.state.lock().await;
        lease.dispatched()?;
        if let Err(error) = self.spawn_locked(&mut state).await {
            lease.complete();
            return Err(error);
        }
        let result = state.worker.as_mut().unwrap().ready().await;
        if result.is_err() {
            state.worker.as_mut().unwrap().stop().await?;
            state.worker = None;
            self.pid.store(0, Ordering::Release);
        }
        lease.complete();
        result
    }

    pub async fn prepare(self: &Arc<Self>, request: LoadRequest) -> Result<Job> {
        if request.model.is_empty()
            || request.model.len() > 512
            || request.model.chars().any(char::is_control)
            || request.digest.len() != 64
            || !request.digest.bytes().all(|b| b.is_ascii_hexdigit())
            || !(512..=1_048_576).contains(&request.context)
            || (request.resident_bytes == 0 && request.peak_bytes != 0)
            || request.peak_bytes < request.resident_bytes
        {
            return Err("managed_load_request_invalid");
        }
        let mut lease = self.reservations.acquire(self.scope()).await?;
        let mut state = self.state.clone().lock_owned().await;
        // Startup itself allocates driver resources, so even cancellation here
        // requires supervisor reconciliation before this scope is reused.
        lease.dispatched()?;
        for attempt in 0..2 {
            if let Err(error) = self.spawn_locked(&mut state).await {
                lease.complete();
                return Err(error);
            }
            let result = self.prepare_locked(&mut state, &request).await;
            match result {
                Ok(address) => {
                    return Ok(Job {
                        address,
                        lease,
                        _state: state,
                    });
                }
                Err(error) => {
                    let previous_residents = state.records.len();
                    let resident_workloads = state
                        .records
                        .values()
                        .map(|record| ResidentWorkload {
                            model_digest: record.workload.model_digest.clone(),
                            context: record.context,
                        })
                        .collect();
                    // Capture evidence while the failed runner still exists.
                    // No unrelated GPU or process is stopped for this check.
                    let before_cleanup = self.memory_sample().await;
                    state.worker.as_mut().unwrap().stop().await?;
                    state.worker = None;
                    self.pid.store(0, Ordering::Release);
                    state.records.clear();
                    let after_cleanup = self.unloaded_samples().await;
                    let retry = self.history.lock().unwrap().record(Failure {
                        model_digest: request.digest.clone(),
                        context: request.context,
                        kind: FailureKind::from_code(error),
                        previous_residents,
                        resident_workloads,
                        before_cleanup,
                        after_cleanup,
                        outcome: Outcome::BaselineUnavailable,
                        isolated_retry: attempt == 0,
                    });
                    if !retry {
                        lease.complete();
                        return Err(error);
                    }
                    // The owned runtime has been reaped; the next attempt
                    // starts empty and rechecks live capacity. User inference
                    // has not been sent, so this does not replay user output.
                }
            }
        }
        unreachable!("the second failed attempt cannot retry")
    }

    pub fn learning(&self) -> History {
        self.history.lock().unwrap().clone()
    }

    async fn memory_sample(&self) -> Option<MemorySample> {
        let gpu = crate::resources::hardware()
            .await
            .ok()?
            .into_iter()
            .find(|g| g.id == self.device_id())?;
        if !gpu.errors.is_empty() {
            return None;
        }
        let memory = gpu.memory?;
        MemorySample::new(memory.total_bytes, memory.free_bytes?)
    }

    async fn unloaded_samples(&self) -> Vec<Option<MemorySample>> {
        let baseline = self.history.lock().unwrap().provisional_unloaded_baseline;
        crate::learning::settle(|| self.memory_sample(), baseline).await
    }

    async fn prepare_locked(&self, state: &mut State, request: &LoadRequest) -> Result<SocketAddr> {
        // The state already owns the process: cancellation during discovery or
        // loading cannot lose the handle needed to stop and reap it.
        state.worker.as_mut().unwrap().ready().await?;
        let mut request = request.clone();
        if request.peak_bytes == 0 {
            if let Some(&(resident, peak)) = state
                .profiles
                .get(&(request.digest.clone(), request.context))
            {
                request.resident_bytes = resident;
                request.peak_bytes = peak;
            } else {
                // First use is an isolated load probe, never a file-size guess.
                let snapshot = self.snapshot(state).await?;
                for resident in snapshot.residents {
                    state.worker.as_mut().unwrap().unload(&resident.id).await?;
                    self.wait_absent(state, &resident.id).await?;
                }
                let snapshot = self.snapshot(state).await?;
                let free = snapshot.devices[0]
                    .free_bytes
                    .ok_or("managed_memory_unknown")?;
                request.peak_bytes = free
                    .checked_sub(self.reserve)
                    // Discovery/driver allocations can move between samples.
                    // Do not make an isolated probe require every free byte.
                    .and_then(|n| n.checked_sub(256 * 1024 * 1024))
                    .filter(|n| *n > 256 * 1024 * 1024)
                    .ok_or("managed_capacity_unavailable")?;
                request.resident_bytes = 1;
            }
        }
        // Fingerprints are only in-process here; persistent learned profiles
        // need pinned runtime/driver/artifact fingerprints before reuse.
        let workload = Workload {
            model_digest: request.digest.clone(),
            runtime_fingerprint: self.id.clone(),
            settings_fingerprint: format!("context:{};parallel:1;default-options", request.context),
        };
        let candidate = Candidate {
            worker: self.id.clone(),
            workload: workload.clone(),
            device_control_verified: true,
            memory: vec![Requirement {
                device: self.device_id(),
                resident_bytes: request.resident_bytes,
                peak_bytes: request.peak_bytes,
            }],
        };
        let mut snapshot = self.snapshot(state).await?;
        let free_before = snapshot.devices[0]
            .free_bytes
            .ok_or("managed_memory_unknown")?;
        // Respect the finite backend ceiling independently of VRAM capacity.
        if snapshot.residents.len() >= self.config.max_loaded_models as usize
            && !snapshot.residents.iter().any(|r| r.workload == workload)
        {
            let old = snapshot
                .residents
                .iter()
                .min_by_key(|r| r.last_used)
                .ok_or("managed_inventory_invalid")?
                .id
                .clone();
            state.worker.as_mut().unwrap().unload(&old).await?;
            self.wait_absent(state, &old).await?;
            snapshot = self.snapshot(state).await?;
        }
        let first = placement::plan(
            &snapshot,
            std::slice::from_ref(&candidate),
            Instant::now(),
            Duration::from_secs(3),
        );
        let plan = match first {
            Decision::Ready(plan) => plan,
            Decision::EvictAndRecheck(plan) => {
                for id in &plan.evict {
                    state.worker.as_mut().unwrap().unload(id).await?;
                    self.wait_absent(state, id).await?;
                }
                // Neither an unload acknowledgment nor a missing /api/ps row
                // establishes actual VRAM release. Wait for a ready plan.
                tokio::time::timeout(Duration::from_secs(10), async {
                    loop {
                        let snapshot = self.snapshot(state).await?;
                        if let Decision::Ready(plan) = placement::plan(
                            &snapshot,
                            std::slice::from_ref(&candidate),
                            Instant::now(),
                            Duration::from_secs(3),
                        ) {
                            return Ok::<_, &'static str>(plan);
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                })
                .await
                .map_err(|_| "managed_memory_release_timeout")??
            }
            Decision::Unavailable(_) => return Err("managed_capacity_unavailable"),
        };
        if plan.reuse.is_none() {
            state
                .worker
                .as_mut()
                .unwrap()
                .load_supervised(&request.model, &request.digest, request.context)
                .await?;
        }
        let running = state.worker.as_mut().unwrap().running().await?;
        let loaded = running
            .models
            .iter()
            .find(|m| {
                m.digest.as_deref() == Some(&request.digest)
                    && m.context_length == Some(u64::from(request.context))
            })
            .ok_or("managed_load_identity_mismatch")?;
        if loaded.residency != crate::resources::ReportedResidency::GpuReported {
            return Err("managed_cpu_spill_detected");
        }
        if loaded
            .vram_bytes
            .is_none_or(|bytes| bytes > request.peak_bytes)
        {
            return Err("managed_peak_estimate_exceeded");
        }
        let sample = crate::resources::snapshot(self.config.address).await;
        let free = sample
            .gpus
            .iter()
            .find(|g| g.id == self.device_id())
            .and_then(|g| g.memory.as_ref())
            .and_then(|m| m.free_bytes)
            .ok_or("managed_memory_unknown")?;
        if free < self.reserve {
            return Err("managed_memory_headroom_exhausted");
        }
        state.sequence = state
            .sequence
            .checked_add(1)
            .ok_or("managed_sequence_exhausted")?;
        let sequence = state.sequence;
        state.records.insert(
            loaded.name.clone(),
            Record {
                workload,
                context: request.context,
                last_used: sequence,
            },
        );
        let resident_bytes = loaded.vram_bytes.ok_or("managed_memory_unknown")?;
        // Load-footprint observation, not a measured inference peak. Additional
        // workspace allowance is separate from the device headroom reserve.
        // These observations stay in RAM until versioned profiling is ready.
        let observed = resident_bytes.max(free_before.saturating_sub(free));
        let peak = observed
            .checked_add(768 * 1024 * 1024)
            .ok_or("managed_peak_invalid")?;
        state
            .profiles
            .entry((request.digest.clone(), request.context))
            .and_modify(|p| {
                p.0 = p.0.max(resident_bytes);
                p.1 = p.1.max(peak);
            })
            .or_insert((resident_bytes, peak));
        Ok(self.config.address)
    }

    async fn wait_absent(&self, state: &mut State, name: &str) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let running = state.worker.as_mut().unwrap().running().await?;
                if !running.models.iter().any(|m| m.name == name) {
                    state.records.remove(name);
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .map_err(|_| "managed_unload_timeout")?
    }

    async fn snapshot(&self, state: &mut State) -> Result<Snapshot> {
        let sampled_at = Instant::now();
        let running = state.worker.as_mut().unwrap().running().await?;
        let hardware = crate::resources::snapshot(self.config.address).await;
        let gpu = hardware
            .gpus
            .into_iter()
            .find(|g| g.id == self.device_id())
            .ok_or("managed_device_missing")?;
        let memory = gpu.memory.ok_or("managed_memory_unknown")?;
        if !gpu.errors.is_empty() {
            return Err("managed_memory_unknown");
        }
        let mut residents = Vec::new();
        state
            .records
            .retain(|name, _| running.models.iter().any(|m| &m.name == name));
        for model in running.models {
            let record = state
                .records
                .get(&model.name)
                .ok_or("managed_unknown_resident")?;
            if model.digest.as_deref() != Some(&record.workload.model_digest)
                || model.context_length != Some(u64::from(record.context))
                || model.residency != crate::resources::ReportedResidency::GpuReported
            {
                return Err("managed_residency_changed");
            }
            residents.push(Resident {
                id: model.name,
                worker: self.id.clone(),
                workload: record.workload.clone(),
                allocations: [(
                    self.device_id(),
                    model.vram_bytes.ok_or("managed_memory_unknown")?,
                )]
                .into(),
                managed: true,
                placement_verified: true,
                idle: true,
                last_used: record.last_used,
            });
        }
        Ok(Snapshot {
            sampled_at,
            devices: vec![Device {
                id: self.device_id(),
                total_bytes: memory.total_bytes,
                free_bytes: memory.free_bytes,
                reserve_bytes: self.reserve,
            }],
            residents,
            busy_devices: Default::default(),
            busy_workers: Default::default(),
        })
    }

    /// Stops only this owned worker after all jobs release their reservation.
    pub async fn stop(&self) -> Result<()> {
        let mut lease = self.reservations.acquire(self.scope()).await?;
        let mut state = self.state.lock().await;
        lease.dispatched()?;
        if let Some(worker) = &mut state.worker {
            worker.stop().await?;
        }
        state.worker = None;
        self.pid.store(0, Ordering::Release);
        state.records.clear();
        lease.complete();
        Ok(())
    }

    pub async fn preference(&self, digest: &str, context: u32) -> u8 {
        let state = self.state.lock().await;
        if state.records.values().any(|r| {
            r.workload.model_digest == digest
                && r.workload.settings_fingerprint
                    == format!("context:{context};parallel:1;default-options")
        }) {
            0
        } else if state.profiles.contains_key(&(digest.to_owned(), context)) {
            1
        } else if state.records.is_empty() {
            2
        } else {
            // An isolated first-use probe would evict this worker's residents.
            // Prefer an empty alternate device before disturbing warm models.
            3
        }
    }

    /// Metadata remains available during a long streamed job. Verify the
    /// listener against the captured owned PID instead of taking the job lock.
    pub async fn running_json(&self) -> Result<serde_json::Value> {
        let pid = self.pid.load(Ordering::Acquire);
        if pid == 0 {
            return Ok(serde_json::json!({"models":[]}));
        }
        crate::systemd::owns_listener(pid, self.config.address)?;
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|_| "worker_client_failed")?;
        let mut response = client
            .get(format!("http://{}/api/ps", self.config.address))
            .send()
            .await
            .map_err(|_| "worker_metadata_unavailable")?;
        if !response.status().is_success() {
            return Err("worker_metadata_unavailable");
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| "worker_metadata_unavailable")?
        {
            if bytes.len() + chunk.len() > 2 * 1024 * 1024 {
                return Err("worker_response_too_large");
            }
            bytes.extend_from_slice(&chunk);
        }
        crate::resources::parse_residency(&bytes)?;
        serde_json::from_slice(&bytes).map_err(|_| "worker_metadata_invalid")
    }
}

pub struct Job {
    address: SocketAddr,
    lease: Reservation,
    // Prevents runtime state changes until upstream execution has ended.
    _state: OwnedMutexGuard<State>,
}
impl Job {
    pub fn address(&self) -> SocketAddr {
        self.address
    }
    pub fn complete(mut self) {
        self.lease.complete();
    }
}
