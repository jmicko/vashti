//! Read-only host/resource observations. No loading, eviction, or placement claims.
use kuno_protocol::API_VERSION;
use serde::Deserialize;
use std::{
    net::SocketAddr,
    sync::{Arc, OnceLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Semaphore;

pub use kuno_protocol::{
    Gpu, Memory, ReportedResidency, Residency, ResidentModel, ResourceSnapshot,
};

#[derive(Deserialize)]
struct Running {
    models: Vec<RunningModel>,
}
#[derive(Deserialize)]
struct RunningModel {
    name: String,
    digest: Option<String>,
    size: Option<u64>,
    size_vram: Option<u64>,
    context_length: Option<u64>,
}

pub(crate) fn parse_residency(bytes: &[u8]) -> Result<Residency, &'static str> {
    let parsed: Running = serde_json::from_slice(bytes).map_err(|_| "ollama_residency_invalid")?;
    if parsed.models.len() > 4096 {
        return Err("ollama_residency_too_large");
    }
    let mut models = Vec::new();
    for model in parsed.models {
        if model.name.is_empty()
            || model.name.len() > 512
            || model.name.chars().any(char::is_control)
            || model
                .digest
                .as_ref()
                .is_some_and(|d| d.len() > 128 || d.chars().any(char::is_control))
        {
            return Err("ollama_residency_invalid");
        }
        let residency = match (model.size, model.size_vram) {
            (Some(total), Some(gpu)) if total > 0 && gpu <= total => {
                if gpu == 0 {
                    ReportedResidency::CpuReported
                } else if gpu == total {
                    ReportedResidency::GpuReported
                } else {
                    ReportedResidency::PartialOffloadReported
                }
            }
            _ => ReportedResidency::Unknown,
        };
        models.push(ResidentModel {
            name: model.name,
            digest: model.digest,
            total_bytes: model.size,
            vram_bytes: model.size_vram,
            context_length: model.context_length,
            residency,
            device_assignment_verified: false,
        });
    }
    Ok(Residency {
        available: true,
        error: None,
        models,
    })
}

async fn residency(upstream: SocketAddr) -> Residency {
    let result = async {
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|_| "ollama_residency_client_failed")?;
        let mut response = client
            .get(format!("http://{upstream}/api/ps"))
            .send()
            .await
            .map_err(|_| "ollama_residency_unavailable")?;
        if !response.status().is_success() {
            return Err("ollama_residency_unavailable");
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| "ollama_residency_unavailable")?
        {
            if bytes.len() + chunk.len() > 2 * 1024 * 1024 {
                return Err("ollama_residency_too_large");
            }
            bytes.extend_from_slice(&chunk);
        }
        parse_residency(&bytes)
    }
    .await;
    result.unwrap_or_else(|error| Residency {
        available: false,
        error: Some(error.into()),
        models: vec![],
    })
}

pub async fn snapshot(upstream: SocketAddr) -> ResourceSnapshot {
    let started = Instant::now();
    let sampled_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let (hardware, ollama) = tokio::join!(hardware(), residency(upstream));
    let (gpus, gpu_errors) = match hardware {
        Ok(gpus) => (gpus, vec![]),
        Err(error) => (vec![], vec![error.into()]),
    };
    ResourceSnapshot {
        api_version: API_VERSION,
        sampled_at_unix_ms,
        collection_duration_ms: started.elapsed().as_millis(),
        scheduling_enabled: false,
        gpus,
        gpu_errors,
        ollama,
    }
}

/// Hardware-only sampling also works after an owned runtime has stopped.
/// Share the probe limit with the control API so a hung driver cannot spawn
/// unbounded probe threads or block recovery indefinitely.
pub(crate) async fn hardware() -> Result<Vec<Gpu>, &'static str> {
    static COLLECTOR: OnceLock<Arc<Semaphore>> = OnceLock::new();
    probe_hardware(
        COLLECTOR
            .get_or_init(|| Arc::new(Semaphore::new(1)))
            .clone(),
        Duration::from_secs(3),
        crate::gpu::collect,
    )
    .await
}

async fn probe_hardware(
    collector: Arc<Semaphore>,
    deadline: Duration,
    probe: impl FnOnce() -> Result<Vec<Gpu>, &'static str> + Send + 'static,
) -> Result<Vec<Gpu>, &'static str> {
    // Keep at most one driver probe alive, even if it hangs past the deadline.
    let permit = collector
        .try_acquire_owned()
        .map_err(|_| "gpu_probe_busy")?;
    let (sender, work) = tokio::sync::oneshot::channel();
    // A standalone thread avoids tokio shutdown waiting for a stuck native
    // library. Its permit prevents blocked workers from accumulating.
    let _worker = std::thread::Builder::new()
        .name("kuno-gpu-probe".into())
        .spawn(move || {
            let _permit = permit;
            let _ = sender.send(probe());
        })
        .map_err(|_| "gpu_probe_failed")?;
    tokio::time::timeout(deadline, work)
        .await
        .map_err(|_| "gpu_probe_timeout")?
        .map_err(|_| "gpu_probe_failed")?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn a_timed_out_driver_keeps_its_slot_until_it_finishes() {
        let collector = Arc::new(Semaphore::new(1));
        let (release, blocked) = std::sync::mpsc::channel();
        let result = probe_hardware(collector.clone(), Duration::from_millis(10), move || {
            blocked.recv().unwrap();
            Ok(vec![])
        })
        .await;
        assert_eq!(result.unwrap_err(), "gpu_probe_timeout");
        let result = probe_hardware(collector.clone(), Duration::from_secs(1), || {
            panic!("must not start a second native probe")
        })
        .await;
        assert_eq!(result.unwrap_err(), "gpu_probe_busy");
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while collector.available_permits() == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        assert!(
            probe_hardware(collector, Duration::from_secs(1), || Ok(vec![]))
                .await
                .is_ok()
        );
    }

    #[test]
    fn residency_distinguishes_cpu_spill_missing_and_inconsistent_counters() {
        let result = parse_residency(
            br#"{"models":[
            {"name":"gpu","size":100,"size_vram":100},
            {"name":"spill","size":100,"size_vram":40},
            {"name":"cpu","size":100,"size_vram":0},
            {"name":"missing","size":100},
            {"name":"invalid","size":100,"size_vram":101},
            {"name":"empty","size":0,"size_vram":0}] }"#,
        )
        .unwrap();
        assert_eq!(
            result
                .models
                .iter()
                .map(|m| &m.residency)
                .collect::<Vec<_>>(),
            vec![
                &ReportedResidency::GpuReported,
                &ReportedResidency::PartialOffloadReported,
                &ReportedResidency::CpuReported,
                &ReportedResidency::Unknown,
                &ReportedResidency::Unknown,
                &ReportedResidency::Unknown
            ]
        );
        assert!(result.models.iter().all(|m| !m.device_assignment_verified));
        assert!(parse_residency(br#"{"models":[{"name":"bad","size":-1}]}"#).is_err());
        assert!(parse_residency(br#"{"error":"private backend detail"}"#).is_err());
    }
}

#[cfg(test)]
mod transport_tests {
    use super::*;
    use axum::{Router, response::Redirect, routing::get};
    #[tokio::test]
    async fn residency_errors_and_redirects_are_classified_without_leaking_payloads() {
        let app = Router::new()
            .route(
                "/api/ps",
                get(|| async { Redirect::temporary("/private-detail") }),
            )
            .route(
                "/private-detail",
                get(|| async { axum::Json(serde_json::json!({"models": []})) }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let report = residency(address).await;
        assert!(!report.available);
        assert_eq!(
            report.error.as_deref(),
            Some("ollama_residency_unavailable")
        );
        assert!(report.models.is_empty());
        server.abort();
    }
}
