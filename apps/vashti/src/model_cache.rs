use std::collections::HashMap;

use sqlx::SqlitePool;
use tokio::sync::{Mutex, Notify};

use crate::{
    auth::service::unix_timestamp,
    backends::service as backends,
    error::ApiError,
    ollama::{self, models::OllamaModel},
};

#[derive(Debug, Clone, Default)]
pub struct CachedBackendModels {
    pub models: Vec<OllamaModel>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelCacheSnapshot {
    pub backends: HashMap<String, CachedBackendModels>,
    pub is_refreshing: bool,
    pub updated_at: Option<i64>,
}

#[derive(Debug, Default)]
struct ModelCacheInner {
    backends: HashMap<String, CachedBackendModels>,
    is_refreshing: bool,
    updated_at: Option<i64>,
}

#[derive(Debug, Default)]
pub struct ModelCache {
    inner: Mutex<ModelCacheInner>,
    notify: Notify,
}

impl ModelCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn snapshot(&self) -> ModelCacheSnapshot {
        let inner = self.inner.lock().await;
        self.snapshot_from_inner(&inner)
    }

    pub async fn refresh_all(
        &self,
        pool: &SqlitePool,
        client: &reqwest::Client,
    ) -> Result<ModelCacheSnapshot, ApiError> {
        let existing = loop {
            let wait = {
                let mut inner = self.inner.lock().await;
                if inner.is_refreshing {
                    Some(self.notify.notified())
                } else {
                    inner.is_refreshing = true;
                    break inner.backends.clone();
                }
            };

            if let Some(wait) = wait {
                wait.await;
            }
        };

        let refresh_result = refresh_backend_models(pool, client, existing).await;
        let mut inner = self.inner.lock().await;

        match refresh_result {
            Ok(backends) => {
                inner.backends = backends;
                inner.updated_at = Some(unix_timestamp());
                inner.is_refreshing = false;
                let snapshot = self.snapshot_from_inner(&inner);
                drop(inner);
                self.notify.notify_waiters();
                Ok(snapshot)
            }
            Err(error) => {
                inner.is_refreshing = false;
                drop(inner);
                self.notify.notify_waiters();
                Err(error)
            }
        }
    }

    fn snapshot_from_inner(&self, inner: &ModelCacheInner) -> ModelCacheSnapshot {
        ModelCacheSnapshot {
            backends: inner.backends.clone(),
            is_refreshing: inner.is_refreshing,
            updated_at: inner.updated_at,
        }
    }
}

async fn refresh_backend_models(
    pool: &SqlitePool,
    client: &reqwest::Client,
    existing: HashMap<String, CachedBackendModels>,
) -> Result<HashMap<String, CachedBackendModels>, ApiError> {
    let mut refreshed = HashMap::new();

    for backend in backends::list_enabled_backends(pool).await? {
        match ollama::client::fetch_models(client, &backend.base_url).await {
            Ok(models) => {
                backends::record_backend_health(pool, &backend.id, "ok", None).await?;
                backends::ensure_model_records(
                    pool,
                    &backend.id,
                    &models
                        .iter()
                        .map(|model| model.name.clone())
                        .collect::<Vec<_>>(),
                )
                .await?;
                refreshed.insert(
                    backend.id,
                    CachedBackendModels {
                        models,
                        last_error: None,
                    },
                );
            }
            Err(error) => {
                let message = error.to_string();
                backends::record_backend_health(pool, &backend.id, "error", Some(&message)).await?;
                tracing::warn!(backend_id = %backend.id, base_url = %backend.base_url, error = %message, "failed to refresh Ollama model cache");
                let mut cached = existing.get(&backend.id).cloned().unwrap_or_default();
                cached.last_error = Some(message);
                refreshed.insert(backend.id, cached);
            }
        }
    }

    Ok(refreshed)
}
