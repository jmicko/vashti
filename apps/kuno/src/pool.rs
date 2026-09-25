//! Managed native Ollama routing. The catalog owner remains the original local
//! Ollama; inference workers share its artifacts but never mutate the catalog.
use crate::{
    managed::{Job, LoadRequest, ManagedWorker},
    reservations::Reservations,
    worker::WorkerConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, net::SocketAddr, path::Path, sync::Arc, time::Duration};

type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolConfig {
    pub workers: Vec<WorkerConfig>,
    pub reserve_bytes: u64,
}

impl PoolConfig {
    pub fn validate(&self, gateway_ports: &[u16]) -> Result<()> {
        if self.workers.is_empty()
            || self.workers.len() > 16
            || self.reserve_bytes < 256 * 1024 * 1024
        {
            return Err("managed_pool_configuration_invalid");
        }
        let mut ports: BTreeSet<_> = gateway_ports.iter().copied().collect();
        let mut devices = BTreeSet::new();
        let mut homes = BTreeSet::new();
        for worker in &self.workers {
            worker.validate()?;
            if !ports.insert(worker.address.port())
                || !devices.insert(worker.device.pci_address())
                || !homes.insert(&worker.home)
                || worker.models != self.workers[0].models
                || worker.context_length != self.workers[0].context_length
            {
                return Err("managed_pool_workers_conflict");
            }
        }
        Ok(())
    }
}

pub struct Pool {
    config: PoolConfig,
    workers: Vec<Arc<ManagedWorker>>,
    reservations: Arc<Reservations>,
    upstream: SocketAddr,
    client: reqwest::Client,
}

impl Pool {
    pub fn new(
        config: PoolConfig,
        upstream: SocketAddr,
        directory: Option<&Path>,
        max_waiting: usize,
        timeout: Duration,
    ) -> Result<Self> {
        let reservations = match directory {
            Some(path) => {
                Reservations::persistent(&path.join("gpu-reservations"), max_waiting, timeout)?
            }
            None => Reservations::new(max_waiting, timeout),
        };
        let mut workers = Vec::new();
        for (i, config_worker) in config.workers.iter().enumerate() {
            workers.push(ManagedWorker::new(
                config_worker.clone(),
                format!("ollama-{i}-{}", config_worker.device.pci_address()),
                config.reserve_bytes,
                reservations.clone(),
            )?);
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|_| "managed_client_failed")?;
        Ok(Self {
            config,
            workers,
            reservations,
            upstream,
            client,
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        for (worker, config) in self.workers.iter().zip(&self.config.workers) {
            std::fs::create_dir_all(&config.home).map_err(|_| "managed_worker_home_unavailable")?;
            if let Err(error) = worker.initialize().await {
                self.stop_all().await?;
                return Err(error);
            }
        }
        Ok(())
    }

    pub fn recovery_required(&self) -> bool {
        self.reservations.recovery_required()
    }

    pub fn learning(&self) -> Vec<Value> {
        self.workers
            .iter()
            .zip(&self.config.workers)
            .map(|(worker, config)| json!({"device": config.device, "evidence": worker.learning()}))
            .collect()
    }

    /// Only routes whose load options are understood may use managed workers.
    /// Ollama keeps responsibility for its compatibility response conversion.
    pub fn supports(path: &str) -> bool {
        matches!(
            path,
            "/api/chat"
                | "/api/generate"
                | "/api/embed"
                | "/api/embeddings"
                | "/v1/chat/completions"
                | "/v1/completions"
                | "/v1/embeddings"
        )
    }

    pub async fn prepare(
        &self,
        path: &str,
        body: &mut Value,
        model: &str,
        digest: &str,
    ) -> Result<Job> {
        let show = self
            .post_metadata("/api/show", &json!({"model":model}))
            .await?;
        let context = request_context(self.config.workers[0].context_length, path, body, &show)?;
        self.release_catalog_residents().await?;
        let mut order = Vec::new();
        for (i, worker) in self.workers.iter().enumerate() {
            order.push((worker.preference(digest, context).await, i));
        }
        order.sort();
        let request = LoadRequest {
            model: model.into(),
            digest: digest.into(),
            context,
            resident_bytes: 0,
            peak_bytes: 0,
        };
        let mut last_error = "managed_no_worker_available";
        for (_, i) in order {
            // Paths are administrator configuration, never request fields.
            std::fs::create_dir_all(&self.config.workers[i].home)
                .map_err(|_| "managed_worker_home_unavailable")?;
            match self.workers[i].prepare(request.clone()).await {
                Ok(job) => return Ok(job),
                Err(error) => {
                    last_error = error;
                    if error == "resource_recovery_required" {
                        self.workers[i].recover().await?;
                        match self.workers[i].prepare(request.clone()).await {
                            Ok(job) => return Ok(job),
                            Err(error) => last_error = error,
                        }
                    }
                }
            }
        }
        Err(last_error)
    }

    /// Caller holds the gateway-wide lease for catalog/opaque operations.
    /// A catalog mutation cannot race cached definitions in resident workers.
    pub async fn stop_all(&self) -> Result<()> {
        for worker in &self.workers {
            if let Err(error) = worker.stop().await {
                if error == "resource_recovery_required" {
                    worker.recover().await?;
                } else {
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// The foreground lease prevents concurrent inference. Verified workers on
    /// other physical GPUs may remain resident; only the image GPU is cleared.
    pub async fn prepare_image(&self, pci_address: &str) -> Result<()> {
        if self.reservations.recovery_required() {
            return Err("resource_recovery_required");
        }
        for (worker, config) in self.workers.iter().zip(&self.config.workers) {
            if config.device.pci_address() == pci_address {
                worker.stop().await?;
            }
        }
        self.release_catalog_residents().await
    }

    pub async fn running(&self) -> Result<Value> {
        let original = self.get_metadata("/api/ps").await?;
        let mut models = original["models"]
            .as_array()
            .ok_or("managed_catalog_invalid")?
            .clone();
        for worker in &self.workers {
            let report = worker.running_json().await?;
            models.extend(
                report["models"]
                    .as_array()
                    .ok_or("worker_metadata_invalid")?
                    .iter()
                    .cloned(),
            );
        }
        Ok(json!({"models":models}))
    }

    async fn release_catalog_residents(&self) -> Result<()> {
        let report = self.get_metadata("/api/ps").await?;
        let models = report["models"]
            .as_array()
            .ok_or("managed_catalog_invalid")?;
        if models.len() > 4096 {
            return Err("managed_catalog_invalid");
        }
        for model in models {
            let name = model
                .get("name")
                .and_then(Value::as_str)
                .ok_or("managed_catalog_invalid")?;
            let response = self
                .post_metadata(
                    "/api/generate",
                    &json!({"model":name,"keep_alive":0,"stream":false}),
                )
                .await?;
            if response.get("done").and_then(Value::as_bool) != Some(true) {
                return Err("managed_catalog_unload_failed");
            }
        }
        if !models.is_empty() {
            tokio::time::timeout(Duration::from_secs(30), async {
                loop {
                    let report = self.get_metadata("/api/ps").await?;
                    if report["models"].as_array().is_some_and(|m| m.is_empty()) {
                        return Ok::<_, &'static str>(());
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            })
            .await
            .map_err(|_| "managed_catalog_unload_timeout")??;
        }
        Ok(())
    }

    async fn get_metadata(&self, path: &str) -> Result<Value> {
        let response = self
            .client
            .get(format!("http://{}{path}", self.upstream))
            .send()
            .await
            .map_err(|_| "managed_catalog_unavailable")?;
        read_metadata(response).await
    }

    async fn post_metadata(&self, path: &str, body: &Value) -> Result<Value> {
        let response = self
            .client
            .post(format!("http://{}{path}", self.upstream))
            .json(body)
            .send()
            .await
            .map_err(|_| "managed_catalog_unavailable")?;
        read_metadata(response).await
    }
}

async fn read_metadata(mut response: reqwest::Response) -> Result<Value> {
    if !response.status().is_success() {
        return Err("managed_model_metadata_unavailable");
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "managed_catalog_unavailable")?
    {
        if bytes.len() + chunk.len() > 4 * 1024 * 1024 {
            return Err("managed_model_metadata_too_large");
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|_| "managed_model_metadata_invalid")
}

// Verified against Ollama v0.34.2 openai/openai.go. Context defaults must match
// the isolated worker's environment and the model's own parameters.
fn request_context(
    mut default_context: u32,
    path: &str,
    body: &mut Value,
    show: &Value,
) -> Result<u32> {
    if !Pool::supports(path) {
        return Err("managed_route_not_supported");
    }
    let object = body.as_object_mut().ok_or("managed_request_invalid")?;
    // Ollama's OpenAI converters ignore native `options`. They use the
    // worker/model default context; never preload using an ignored override.
    let mut compatibility_options = serde_json::Map::new();
    let object = if path.starts_with("/api/") {
        object
    } else {
        &mut compatibility_options
    };
    let options = object.entry("options").or_insert_with(|| json!({}));
    if options.is_null() {
        *options = json!({});
    }
    let options = options.as_object_mut().ok_or("managed_options_invalid")?;
    // Memory-affecting overrides require their own load identity/preload
    // mapping. Reject rather than preload one shape and execute another.
    if options.keys().any(|k| {
        matches!(
            k.as_str(),
            "num_gpu"
                | "main_gpu"
                | "num_batch"
                | "use_mmap"
                | "use_mlock"
                | "numa"
                | "f16_kv"
                | "low_vram"
                | "draft_num_predict"
        )
    }) {
        return Err("managed_memory_option_not_supported");
    }
    let explicit = options.get("num_ctx").filter(|v| !v.is_null());

    if let Some(parameters) = show.get("parameters").and_then(Value::as_str) {
        for line in parameters.lines() {
            let mut words = line.split_whitespace();
            if words.next() == Some("num_ctx") {
                default_context = words
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("managed_model_context_invalid")?;
            }
        }
    }
    let maximum = show
        .get("model_info")
        .and_then(Value::as_object)
        .and_then(|info| {
            let arch = info.get("general.architecture")?.as_str()?;
            info.get(&format!("{arch}.context_length"))?.as_u64()
        });
    let context = match explicit {
        Some(value) => value
            .as_u64()
            .filter(|n| (512..=1_048_576).contains(n))
            .ok_or("managed_requested_context_invalid")?,
        None => maximum.map_or(u64::from(default_context), |n| {
            u64::from(default_context).min(n)
        }),
    };
    if maximum.is_some_and(|max| context > max) {
        return Err("managed_requested_context_exceeds_model_limit");
    }
    let context = u32::try_from(context).map_err(|_| "managed_requested_context_invalid")?;
    options.insert("num_ctx".into(), json!(context));
    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compatibility_uses_actual_backend_defaults_without_rewriting_payload() {
        let show = json!({"model_info":{"general.architecture":"test","test.context_length":2048}});
        for path in ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"] {
            let mut body = json!({"model":"test","options":{"num_ctx":16384,"num_gpu":0},"max_tokens":12,"stream":true});
            let original = body.clone();
            assert_eq!(request_context(4096, path, &mut body, &show), Ok(2048));
            assert_eq!(body, original);
        }
    }
    #[test]
    fn native_options_and_model_context_are_checked_before_loading() {
        let show = json!({"parameters":"num_ctx 8192\ntemperature 0.4", "model_info":{"general.architecture":"test","test.context_length":16384}});
        let mut body = json!({"model":"test"});
        assert_eq!(
            request_context(4096, "/api/chat", &mut body, &show),
            Ok(8192)
        );
        assert_eq!(body["options"]["num_ctx"], 8192);
        for (options, expected) in [
            (
                json!({"num_ctx":32768}),
                "managed_requested_context_exceeds_model_limit",
            ),
            (json!({"num_ctx":0}), "managed_requested_context_invalid"),
            (
                json!({"num_ctx":"8192"}),
                "managed_requested_context_invalid",
            ),
            (json!({"num_gpu":0}), "managed_memory_option_not_supported"),
        ] {
            assert_eq!(
                request_context(4096, "/api/chat", &mut json!({"options":options}), &show),
                Err(expected)
            );
        }
    }
}
