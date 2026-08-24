use std::{collections::HashMap, io, sync::Arc};

use futures_util::TryStreamExt;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    sync::Mutex,
};
use tokio_util::io::StreamReader;
use uuid::Uuid;

use crate::{
    app_state::AppState, auth::service::unix_timestamp, backends::service as backends,
    error::ApiError, ollama,
};

#[derive(Clone, Debug, Serialize)]
pub struct ModelPullStatus {
    pub id: Option<String>,
    pub backend_id: String,
    pub model: Option<String>,
    pub state: &'static str,
    pub status: Option<String>,
    pub completed: Option<u64>,
    pub total: Option<u64>,
    pub error: Option<String>,
    pub updated_at: Option<i64>,
}

impl ModelPullStatus {
    fn idle(backend_id: &str) -> Self {
        Self {
            id: None,
            backend_id: backend_id.to_string(),
            model: None,
            state: "idle",
            status: None,
            completed: None,
            total: None,
            error: None,
            updated_at: None,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self.state, "queued" | "running" | "refreshing")
    }
}

#[derive(Debug, Deserialize)]
struct OllamaPullProgress {
    status: Option<String>,
    error: Option<String>,
    completed: Option<u64>,
    total: Option<u64>,
}

#[derive(Debug, Default)]
pub struct ModelPullManager {
    jobs: Mutex<HashMap<String, ModelPullStatus>>,
}

impl ModelPullManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn status(&self, backend_id: &str) -> ModelPullStatus {
        self.jobs
            .lock()
            .await
            .get(backend_id)
            .cloned()
            .unwrap_or_else(|| ModelPullStatus::idle(backend_id))
    }

    pub async fn start(
        self: &Arc<Self>,
        state: AppState,
        backend_id: String,
        input: String,
    ) -> Result<ModelPullStatus, ApiError> {
        let model = parse_model_pull_input(&input)?;
        let backend = backends::get_enabled_backend(&state.db, &backend_id)
            .await?
            .ok_or_else(|| {
                ApiError::not_found(
                    "backend_not_found",
                    "Choose an enabled Ollama backend before pulling a model",
                )
            })?;

        let mut jobs = self.jobs.lock().await;
        if let Some(existing) = jobs.get(&backend_id).filter(|job| job.is_active()) {
            return Err(ApiError::conflict(
                "model_pull_in_progress",
                format!(
                    "{} is already pulling a model on this backend",
                    existing.model.as_deref().unwrap_or("Vashti")
                ),
            ));
        }

        let job = ModelPullStatus {
            id: Some(Uuid::new_v4().to_string()),
            backend_id: backend_id.clone(),
            model: Some(model.clone()),
            state: "queued",
            status: Some("Waiting for Ollama".to_string()),
            completed: None,
            total: None,
            error: None,
            updated_at: Some(unix_timestamp()),
        };
        jobs.insert(backend_id.clone(), job.clone());
        drop(jobs);

        let manager = Arc::clone(self);
        tokio::spawn(async move {
            manager
                .run_job(state, backend.id, backend.base_url, model)
                .await;
        });
        Ok(job)
    }

    async fn run_job(&self, state: AppState, backend_id: String, base_url: String, model: String) {
        self.update(&backend_id, |job| {
            job.state = "running";
            job.status = Some("Connecting to Ollama".to_string());
        })
        .await;

        let result = self
            .consume_pull(&state.http_client, &backend_id, &base_url, &model)
            .await;
        if let Err(error) = result {
            tracing::warn!(backend_id, model, %error, "Ollama model pull failed");
            self.update(&backend_id, |job| {
                job.state = "failed";
                job.status = Some("Pull failed".to_string());
                job.error = Some(error);
            })
            .await;
            return;
        }

        self.update(&backend_id, |job| {
            job.state = "refreshing";
            job.status = Some("Refreshing the model list".to_string());
            job.error = None;
        })
        .await;
        match state
            .model_cache
            .refresh_all(&state.db, &state.http_client)
            .await
        {
            Ok(_) => {
                self.update(&backend_id, |job| {
                    job.state = "succeeded";
                    job.status = Some("Model ready".to_string());
                    job.error = None;
                    if let Some(total) = job.total {
                        job.completed = Some(total);
                    }
                })
                .await;
            }
            Err(error) => {
                let message = error.message().to_string();
                self.update(&backend_id, |job| {
                    job.state = "failed";
                    job.status = Some("Model downloaded, but refresh failed".to_string());
                    job.error = Some(message);
                })
                .await;
            }
        }
    }

    async fn consume_pull(
        &self,
        client: &reqwest::Client,
        backend_id: &str,
        base_url: &str,
        model: &str,
    ) -> Result<(), String> {
        let response = ollama::client::pull_model_stream(client, base_url, model)
            .await
            .map_err(|error| format!("Ollama rejected the pull: {error}"))?;
        let stream = response
            .bytes_stream()
            .map_err(|error| io::Error::other(error.to_string()));
        let mut lines = BufReader::new(StreamReader::new(stream)).lines();
        let mut saw_success = false;

        while let Some(line) = lines
            .next_line()
            .await
            .map_err(|error| format!("failed to read Ollama progress: {error}"))?
        {
            if line.trim().is_empty() {
                continue;
            }
            let progress: OllamaPullProgress = serde_json::from_str(&line)
                .map_err(|error| format!("Ollama returned invalid progress data: {error}"))?;
            if let Some(error) = progress.error.filter(|error| !error.trim().is_empty()) {
                return Err(error);
            }
            saw_success |= progress.status.as_deref() == Some("success");
            self.update(backend_id, |job| {
                if let Some(status) = progress.status.clone() {
                    job.status = Some(humanize_pull_status(&status));
                }
                if progress.completed.is_some() {
                    job.completed = progress.completed;
                }
                if progress.total.is_some() {
                    job.total = progress.total;
                }
            })
            .await;
        }

        saw_success
            .then_some(())
            .ok_or_else(|| "Ollama ended the pull before confirming success".to_string())
    }

    async fn update(&self, backend_id: &str, update: impl FnOnce(&mut ModelPullStatus)) {
        let mut jobs = self.jobs.lock().await;
        if let Some(job) = jobs.get_mut(backend_id) {
            update(job);
            job.updated_at = Some(unix_timestamp());
        }
    }
}

pub fn parse_model_pull_input(input: &str) -> Result<String, ApiError> {
    let trimmed = input.trim().trim_matches('`').trim();
    let parts = trimmed.split_whitespace().collect::<Vec<_>>();
    let model = match parts.as_slice() {
        [model] => *model,
        ["ollama", "pull", model] => *model,
        _ => {
            return Err(ApiError::bad_request(
                "invalid_model_pull",
                "Enter a model name or a command like: ollama pull gemma3:4b",
            ));
        }
    };
    if model.is_empty()
        || model.len() > 512
        || model.starts_with('-')
        || !model
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "._-/:".contains(character))
    {
        return Err(ApiError::bad_request(
            "invalid_model_name",
            "The model name contains unsupported characters",
        ));
    }
    Ok(model.to_string())
}

fn humanize_pull_status(status: &str) -> String {
    let mut characters = status.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => "Working".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use axum::{Router, routing::post};

    use super::*;

    async fn test_pull_server(body: &'static str) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route(
                    "/api/pull",
                    post(move || async move {
                        (
                            [
                                ("content-type", "application/x-ndjson"),
                                ("cache-control", "no-store"),
                            ],
                            body,
                        )
                    }),
                ),
            )
            .await
            .unwrap();
        });
        (format!("http://{address}"), server)
    }

    fn queued_job(backend_id: &str) -> ModelPullStatus {
        ModelPullStatus {
            id: Some("test-job".to_string()),
            backend_id: backend_id.to_string(),
            model: Some("gemma3:4b".to_string()),
            state: "running",
            status: None,
            completed: None,
            total: None,
            error: None,
            updated_at: None,
        }
    }

    #[test]
    fn model_pull_input_accepts_names_and_copied_commands() {
        assert_eq!(parse_model_pull_input("gemma3:4b").unwrap(), "gemma3:4b");
        assert_eq!(
            parse_model_pull_input("ollama pull hf.co/example/model:Q4_K_M").unwrap(),
            "hf.co/example/model:Q4_K_M"
        );
        assert_eq!(
            parse_model_pull_input("`ollama pull qwen3:8b`").unwrap(),
            "qwen3:8b"
        );
    }

    #[test]
    fn model_pull_input_rejects_shell_syntax_and_flags() {
        for input in [
            "ollama pull model;reboot",
            "ollama pull --insecure model",
            "$(reboot)",
            "-dangerous",
            "ollama run model",
        ] {
            assert!(parse_model_pull_input(input).is_err(), "accepted {input}");
        }
    }

    #[tokio::test]
    async fn consumes_streamed_ollama_progress() {
        let (base_url, server) = test_pull_server(
            "{\"status\":\"pulling manifest\"}\n{\"status\":\"downloading\",\"completed\":524288,\"total\":1048576}\n{\"status\":\"success\"}\n",
        )
        .await;
        let manager = ModelPullManager::new();
        manager
            .jobs
            .lock()
            .await
            .insert("backend".to_string(), queued_job("backend"));

        manager
            .consume_pull(&reqwest::Client::new(), "backend", &base_url, "gemma3:4b")
            .await
            .unwrap();

        let status = manager.status("backend").await;
        assert_eq!(status.status.as_deref(), Some("Success"));
        assert_eq!(status.completed, Some(524_288));
        assert_eq!(status.total, Some(1_048_576));
        server.abort();
    }

    #[tokio::test]
    async fn rejects_a_pull_stream_without_success() {
        let (base_url, server) = test_pull_server("{\"status\":\"pulling manifest\"}\n").await;
        let manager = ModelPullManager::new();
        manager
            .jobs
            .lock()
            .await
            .insert("backend".to_string(), queued_job("backend"));

        let error = manager
            .consume_pull(&reqwest::Client::new(), "backend", &base_url, "gemma3:4b")
            .await
            .unwrap_err();

        assert!(error.contains("before confirming success"));
        server.abort();
    }
}
