use std::{
    collections::{HashMap, HashSet},
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};

use sqlx::{Row, SqlitePool};
use tokio::sync::{Mutex, Notify};

use crate::{
    auth::service::unix_timestamp,
    memories::{models::MemorySummaryResponse, service as memory_service},
    ollama,
    vector_index::{VectorIndex, normalize},
};

const MIN_CACHE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CACHE_BYTES: usize = 512 * 1024 * 1024;
const FALLBACK_CACHE_BYTES: usize = 128 * 1024 * 1024;
const FUSION_K: f64 = 60.0;

#[derive(Clone, Debug)]
struct ActiveConfig {
    backend_id: String,
    base_url: String,
    model: String,
}

#[derive(Clone, Debug)]
struct EmbeddingJob {
    memory_id: String,
    user_id: String,
    memory_version_id: String,
    content: String,
}

#[derive(Clone, Debug)]
struct CachedMemory {
    memory_id: String,
}

#[derive(Debug)]
struct CachedUserVectors {
    revision: i64,
    backend_id: String,
    model: String,
    index: VectorIndex<CachedMemory>,
    bytes: usize,
}

#[derive(Debug)]
struct CachedUserEntry {
    vectors: Arc<CachedUserVectors>,
    last_used: u64,
}

#[derive(Debug, Default)]
struct EmbeddingCache {
    users: HashMap<String, CachedUserEntry>,
    total_bytes: usize,
    clock: u64,
}

#[derive(Clone, Debug)]
struct SemanticMatch {
    memory_id: String,
    score: f32,
}

#[derive(Debug)]
enum JobResult {
    Indexed,
    NoWork,
}

#[derive(Debug)]
pub struct MemoryRetrieval {
    pool: SqlitePool,
    cache: Mutex<EmbeddingCache>,
    cache_budget_bytes: usize,
    notify: Notify,
}

impl MemoryRetrieval {
    pub fn new(pool: SqlitePool) -> Self {
        let cache_budget_bytes = adaptive_cache_bytes();
        tracing::info!(
            cache_budget_mib = cache_budget_bytes / (1024 * 1024),
            "configured memory vector cache"
        );
        Self {
            pool,
            cache: Mutex::new(EmbeddingCache::default()),
            cache_budget_bytes,
            notify: Notify::new(),
        }
    }

    pub fn wake(&self) {
        self.notify.notify_one();
    }

    pub async fn run(self: Arc<Self>, client: reqwest::Client) {
        loop {
            if let Err(error) = self.repair_index().await {
                tracing::warn!(%error, "failed to repair the memory embedding queue");
            }
            loop {
                match self.process_next_job(&client).await {
                    Ok(JobResult::Indexed) => continue,
                    Ok(JobResult::NoWork) => break,
                    Err(error) => {
                        tracing::warn!(%error, "failed to index a memory");
                        break;
                    }
                }
            }
            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_secs(30)) => {}
            }
        }
    }

    pub async fn rebuild_all(&self) -> Result<(), sqlx::Error> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("DELETE FROM memory_embeddings")
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM memory_embedding_jobs")
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            r#"
            INSERT INTO memory_embedding_jobs (
                memory_id, user_id, memory_version_id, requested_at,
                attempts, last_attempt_at, last_error
            )
            SELECT id, user_id, current_version_id, updated_at, 0, NULL, NULL
            FROM memories
            WHERE deleted_at IS NULL
              AND EXISTS (
                  SELECT 1 FROM app_settings
                  WHERE id = 1 AND notes_semantic_search_enabled = 1
              )
            "#,
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            r#"
            INSERT INTO memory_embedding_revisions (user_id, revision)
            SELECT DISTINCT user_id, 1 FROM memories
            WHERE 1 = 1
            ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1
            "#,
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        *self.cache.lock().await = EmbeddingCache::default();
        self.wake();
        Ok(())
    }

    pub async fn hybrid_search(
        &self,
        client: &reqwest::Client,
        user_id: &str,
        model_key: &str,
        query: &str,
        limit: i64,
    ) -> Result<Vec<MemorySummaryResponse>, crate::error::ApiError> {
        let limit = limit.clamp(1, 10) as usize;
        let keyword =
            memory_service::search_memories_for_model(&self.pool, user_id, model_key, query, 10)
                .await?;
        let semantic = match self
            .semantic_search(client, user_id, model_key, query, 10)
            .await
        {
            Ok(matches) => matches,
            Err(error) => {
                tracing::warn!(%error, "semantic memory search unavailable; using keywords");
                return Ok(keyword.into_iter().take(limit).collect());
            }
        };
        if semantic.is_empty() {
            return Ok(keyword.into_iter().take(limit).collect());
        }

        let mut scores = HashMap::<String, f64>::new();
        let mut summaries = HashMap::new();
        for (index, memory) in keyword.into_iter().enumerate() {
            *scores.entry(memory.id.clone()).or_default() += 1.15 / (FUSION_K + index as f64 + 1.0);
            summaries.insert(memory.id.clone(), memory);
        }
        for (index, memory) in semantic.into_iter().enumerate() {
            *scores.entry(memory.memory_id).or_default() += 1.0 / (FUSION_K + index as f64 + 1.0);
        }
        let mut ranked = scores.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        let mut results = Vec::with_capacity(limit);
        for (memory_id, _) in ranked {
            if let Some(summary) = summaries.remove(&memory_id) {
                results.push(summary);
            } else if let Ok(memory) =
                memory_service::get_memory_for_model(&self.pool, user_id, &memory_id, model_key)
                    .await
            {
                results.push(MemorySummaryResponse {
                    id: memory.id,
                    excerpt: truncate(&memory.current_version.content, 420),
                    current_version_id: memory.current_version.id,
                    current_version_number: memory.current_version.version_number,
                    model_scope: memory.model_scope,
                    deleted_at: memory.deleted_at,
                    created_at: memory.created_at,
                    updated_at: memory.updated_at,
                });
            }
            if results.len() == limit {
                break;
            }
        }
        Ok(results)
    }

    async fn semantic_search(
        &self,
        client: &reqwest::Client,
        user_id: &str,
        model_key: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SemanticMatch>, String> {
        let started = Instant::now();
        let Some(config) = self.active_config().await? else {
            return Ok(Vec::new());
        };
        let allowed = self.allowed_memory_ids(user_id, model_key).await?;
        if allowed.is_empty() {
            return Ok(Vec::new());
        }
        let response = ollama::client::embed(
            client,
            &config.base_url,
            &config.model,
            &[query.to_string()],
        )
        .await
        .map_err(|error| format!("query embedding failed: {error}"))?;
        let query_vector = response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| "embedding backend returned no query vector".to_string())?;
        validate_vector(&query_vector, None)?;
        let revision = self.user_revision(user_id).await?;
        let vectors = self.ensure_user_cache(user_id, revision, &config).await?;
        let mut matches = Vec::new();
        vectors
            .index
            .for_each_score(&query_vector, |_, memory, score| {
                if allowed.contains(&memory.memory_id) {
                    matches.push(SemanticMatch {
                        memory_id: memory.memory_id.clone(),
                        score,
                    });
                }
            })?;
        matches.sort_unstable_by(|left, right| right.score.total_cmp(&left.score));
        matches.truncate(limit);
        tracing::debug!(
            user_id,
            indexed_memories = vectors.index.len(),
            allowed_memories = allowed.len(),
            result_count = matches.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "completed semantic memory search"
        );
        Ok(matches)
    }

    async fn ensure_user_cache(
        &self,
        user_id: &str,
        revision: i64,
        config: &ActiveConfig,
    ) -> Result<Arc<CachedUserVectors>, String> {
        {
            let mut cache = self.cache.lock().await;
            cache.clock = cache.clock.wrapping_add(1);
            let clock = cache.clock;
            if let Some(entry) = cache.users.get_mut(user_id).filter(|entry| {
                entry.vectors.revision == revision
                    && entry.vectors.backend_id == config.backend_id
                    && entry.vectors.model == config.model
            }) {
                entry.last_used = clock;
                return Ok(Arc::clone(&entry.vectors));
            }
        }

        let rows = sqlx::query(
            r#"
            SELECT embeddings.memory_id, embeddings.vector
            FROM memory_embeddings embeddings
            JOIN memories m ON m.id = embeddings.memory_id
            WHERE embeddings.user_id = ?
              AND embeddings.memory_version_id = m.current_version_id
              AND m.deleted_at IS NULL
              AND embeddings.embedding_backend_id = ?
              AND embeddings.embedding_model = ?
            ORDER BY m.updated_at DESC
            "#,
        )
        .bind(user_id)
        .bind(&config.backend_id)
        .bind(&config.model)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("failed to load memory vectors: {error}"))?;
        let mut entries = Vec::with_capacity(rows.len());
        let mut metadata_bytes = 0usize;
        for row in rows {
            let memory_id: String = row.try_get("memory_id").map_err(db_value_error)?;
            let vector_bytes: Vec<u8> = row.try_get("vector").map_err(db_value_error)?;
            metadata_bytes = metadata_bytes
                .saturating_add(memory_id.capacity())
                .saturating_add(size_of::<CachedMemory>());
            entries.push((
                CachedMemory { memory_id },
                vector_from_bytes(&vector_bytes)?,
            ));
        }
        let index = VectorIndex::build(entries)?;
        let bytes = metadata_bytes.saturating_add(index.allocated_vector_bytes());
        let loaded = Arc::new(CachedUserVectors {
            revision,
            backend_id: config.backend_id.clone(),
            model: config.model.clone(),
            index,
            bytes,
        });

        let mut cache = self.cache.lock().await;
        cache.clock = cache.clock.wrapping_add(1);
        let last_used = cache.clock;
        if let Some(previous) = cache.users.remove(user_id) {
            cache.total_bytes = cache.total_bytes.saturating_sub(previous.vectors.bytes);
        }
        cache.total_bytes = cache.total_bytes.saturating_add(bytes);
        cache.users.insert(
            user_id.to_string(),
            CachedUserEntry {
                vectors: Arc::clone(&loaded),
                last_used,
            },
        );
        while cache.total_bytes > self.cache_budget_bytes && cache.users.len() > 1 {
            let Some(oldest) = cache
                .users
                .iter()
                .filter(|(cached_user, _)| cached_user.as_str() != user_id)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(user, _)| user.clone())
            else {
                break;
            };
            if let Some(entry) = cache.users.remove(&oldest) {
                cache.total_bytes = cache.total_bytes.saturating_sub(entry.vectors.bytes);
            }
        }
        Ok(loaded)
    }

    async fn allowed_memory_ids(
        &self,
        user_id: &str,
        model_key: &str,
    ) -> Result<HashSet<String>, String> {
        sqlx::query_scalar(
            r#"
            SELECT m.id
            FROM memories m
            LEFT JOIN user_memory_settings settings ON settings.user_id = m.user_id
            WHERE m.user_id = ?
              AND m.deleted_at IS NULL
              AND COALESCE(settings.allow_model_read, 1) = 1
              AND (
                  m.all_models = 1
                  OR EXISTS (
                      SELECT 1 FROM memory_model_scopes scopes
                      WHERE scopes.memory_id = m.id AND scopes.model_key = ?
                  )
              )
            "#,
        )
        .bind(user_id)
        .bind(model_key)
        .fetch_all(&self.pool)
        .await
        .map(|ids| ids.into_iter().collect())
        .map_err(|error| format!("failed to apply memory permissions: {error}"))
    }

    async fn user_revision(&self, user_id: &str) -> Result<i64, String> {
        sqlx::query_scalar("SELECT revision FROM memory_embedding_revisions WHERE user_id = ?")
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map(|revision| revision.unwrap_or(0))
            .map_err(|error| format!("failed to read memory index revision: {error}"))
    }

    async fn process_next_job(&self, client: &reqwest::Client) -> Result<JobResult, String> {
        let Some(config) = self.active_config().await? else {
            return Ok(JobResult::NoWork);
        };
        let now = unix_timestamp();
        let row = sqlx::query(
            r#"
            SELECT jobs.memory_id, jobs.user_id, jobs.memory_version_id, versions.content
            FROM memory_embedding_jobs jobs
            JOIN memories m ON m.id = jobs.memory_id
            JOIN memory_versions versions ON versions.id = jobs.memory_version_id
            WHERE m.deleted_at IS NULL
              AND m.current_version_id = jobs.memory_version_id
              AND (
                  jobs.last_attempt_at IS NULL
                  OR jobs.last_attempt_at <= ? - MIN(300, MAX(15, jobs.attempts * 15))
              )
            ORDER BY jobs.requested_at ASC
            LIMIT 1
            "#,
        )
        .bind(now)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("failed to read memory embedding queue: {error}"))?;
        let Some(row) = row else {
            return Ok(JobResult::NoWork);
        };
        let job = EmbeddingJob {
            memory_id: row.try_get("memory_id").map_err(db_value_error)?,
            user_id: row.try_get("user_id").map_err(db_value_error)?,
            memory_version_id: row.try_get("memory_version_id").map_err(db_value_error)?,
            content: row.try_get("content").map_err(db_value_error)?,
        };
        let response = match ollama::client::embed(
            client,
            &config.base_url,
            &config.model,
            std::slice::from_ref(&job.content),
        )
        .await
        {
            Ok(response) => response,
            Err(error) => {
                let message = format!("embedding request failed: {error}");
                self.record_failure(&job.memory_id, &message).await;
                return Err(message);
            }
        };
        let mut vector = response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| "embedding backend returned no memory vector".to_string())?;
        validate_vector(&vector, None)?;
        normalize(&mut vector)?;

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("failed to save memory embedding: {error}"))?;
        let still_current: i64 = sqlx::query_scalar(
            r#"
            SELECT EXISTS(
                SELECT 1 FROM memories m
                JOIN app_settings settings ON settings.id = 1
                WHERE m.id = ? AND m.current_version_id = ? AND m.deleted_at IS NULL
                  AND settings.notes_semantic_search_enabled = 1
                  AND settings.notes_embedding_backend_id = ?
                  AND settings.notes_embedding_model = ?
            )
            "#,
        )
        .bind(&job.memory_id)
        .bind(&job.memory_version_id)
        .bind(&config.backend_id)
        .bind(&config.model)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("failed to verify memory embedding source: {error}"))?;
        if still_current == 0 {
            tx.rollback().await.ok();
            return Ok(JobResult::Indexed);
        }
        sqlx::query(
            r#"
            INSERT INTO memory_embeddings (
                memory_id, user_id, memory_version_id, embedding_backend_id,
                embedding_model, embedding_dimensions, vector, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                user_id = excluded.user_id,
                memory_version_id = excluded.memory_version_id,
                embedding_backend_id = excluded.embedding_backend_id,
                embedding_model = excluded.embedding_model,
                embedding_dimensions = excluded.embedding_dimensions,
                vector = excluded.vector,
                indexed_at = excluded.indexed_at
            "#,
        )
        .bind(&job.memory_id)
        .bind(&job.user_id)
        .bind(&job.memory_version_id)
        .bind(&config.backend_id)
        .bind(&config.model)
        .bind(vector.len() as i64)
        .bind(vector_to_bytes(&vector))
        .bind(now)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("failed to insert memory embedding: {error}"))?;
        sqlx::query(
            "DELETE FROM memory_embedding_jobs WHERE memory_id = ? AND memory_version_id = ?",
        )
        .bind(&job.memory_id)
        .bind(&job.memory_version_id)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("failed to finish memory embedding job: {error}"))?;
        sqlx::query(
            r#"
            INSERT INTO memory_embedding_revisions (user_id, revision)
            VALUES (?, 1)
            ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1
            "#,
        )
        .bind(&job.user_id)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("failed to publish memory index revision: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("failed to commit memory embedding: {error}"))?;
        Ok(JobResult::Indexed)
    }

    async fn repair_index(&self) -> Result<(), sqlx::Error> {
        let Some(config) = self.active_config().await.map_err(sqlx::Error::Protocol)? else {
            return Ok(());
        };
        sqlx::query(
            r#"
            INSERT INTO memory_embedding_jobs (
                memory_id, user_id, memory_version_id, requested_at,
                attempts, last_attempt_at, last_error
            )
            SELECT m.id, m.user_id, m.current_version_id, m.updated_at, 0, NULL, NULL
            FROM memories m
            WHERE m.deleted_at IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM memory_embeddings embeddings
                  WHERE embeddings.memory_id = m.id
                    AND embeddings.memory_version_id = m.current_version_id
                    AND embeddings.embedding_backend_id = ?
                    AND embeddings.embedding_model = ?
              )
            ON CONFLICT(memory_id) DO UPDATE SET
                user_id = excluded.user_id,
                memory_version_id = excluded.memory_version_id,
                requested_at = excluded.requested_at
            "#,
        )
        .bind(config.backend_id)
        .bind(config.model)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn active_config(&self) -> Result<Option<ActiveConfig>, String> {
        let row = sqlx::query(
            r#"
            SELECT settings.notes_semantic_search_enabled,
                   settings.notes_embedding_backend_id,
                   settings.notes_embedding_model,
                   backends.base_url
            FROM app_settings settings
            LEFT JOIN ollama_backends backends
              ON backends.id = settings.notes_embedding_backend_id
             AND backends.is_enabled = 1
            WHERE settings.id = 1
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|error| format!("failed to read knowledge-search settings: {error}"))?;
        if row
            .try_get::<i64, _>("notes_semantic_search_enabled")
            .map_err(db_value_error)?
            == 0
        {
            return Ok(None);
        }
        let backend_id: Option<String> = row
            .try_get("notes_embedding_backend_id")
            .map_err(db_value_error)?;
        let model: Option<String> = row
            .try_get("notes_embedding_model")
            .map_err(db_value_error)?;
        let base_url: Option<String> = row.try_get("base_url").map_err(db_value_error)?;
        match (backend_id, base_url, model) {
            (Some(backend_id), Some(base_url), Some(model)) if !model.trim().is_empty() => {
                Ok(Some(ActiveConfig {
                    backend_id,
                    base_url,
                    model,
                }))
            }
            _ => Err("knowledge search has no enabled embedding backend and model".to_string()),
        }
    }

    async fn record_failure(&self, memory_id: &str, error: &str) {
        let now = unix_timestamp();
        if let Err(db_error) = sqlx::query(
            r#"
            UPDATE memory_embedding_jobs
            SET attempts = attempts + 1, last_attempt_at = ?, last_error = ?
            WHERE memory_id = ?
            "#,
        )
        .bind(now)
        .bind(truncate(error, 500))
        .bind(memory_id)
        .execute(&self.pool)
        .await
        {
            tracing::warn!(?db_error, "failed to record memory embedding error");
        }
    }
}

pub async fn index_status(pool: &SqlitePool) -> Result<(i64, i64, Option<String>), sqlx::Error> {
    let indexed = sqlx::query_scalar("SELECT COUNT(*) FROM memory_embeddings")
        .fetch_one(pool)
        .await?;
    let pending = sqlx::query_scalar("SELECT COUNT(*) FROM memory_embedding_jobs")
        .fetch_one(pool)
        .await?;
    let last_error = sqlx::query_scalar(
        r#"
        SELECT last_error FROM memory_embedding_jobs
        WHERE last_error IS NOT NULL
        ORDER BY last_attempt_at DESC LIMIT 1
        "#,
    )
    .fetch_optional(pool)
    .await?
    .flatten();
    Ok((indexed, pending, last_error))
}

fn validate_vector(vector: &[f32], expected_dimensions: Option<usize>) -> Result<(), String> {
    if vector.is_empty() {
        return Err("embedding backend returned an empty vector".to_string());
    }
    if expected_dimensions.is_some_and(|dimensions| dimensions != vector.len()) {
        return Err("embedding backend returned inconsistent vector dimensions".to_string());
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("embedding backend returned a non-finite vector".to_string());
    }
    Ok(())
}

fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn vector_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(size_of::<f32>()) {
        return Err("stored memory embedding has an invalid byte length".to_string());
    }
    let vector = bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    validate_vector(&vector, None)?;
    Ok(vector)
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut result = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        result.push('…');
    }
    result
}

fn db_value_error(error: sqlx::Error) -> String {
    format!("invalid memory index database value: {error}")
}

fn adaptive_cache_bytes() -> usize {
    physical_memory_bytes()
        .map(|bytes| bytes / 64)
        .unwrap_or(FALLBACK_CACHE_BYTES)
        .clamp(MIN_CACHE_BYTES, MAX_CACHE_BYTES)
}

#[cfg(unix)]
fn physical_memory_bytes() -> Option<usize> {
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages <= 0 || page_size <= 0 {
        return None;
    }
    usize::try_from(pages)
        .ok()?
        .checked_mul(usize::try_from(page_size).ok()?)
}

#[cfg(not(unix))]
fn physical_memory_bytes() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use axum::{Json, Router, routing::post};
    use serde::Deserialize;
    use serde_json::json;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    use super::*;
    use crate::{
        memories::models::{CreateMemoryRequest, MemoryModelScope, UpdateMemoryRequest},
        startup,
    };

    async fn test_pool() -> SqlitePool {
        let options = SqliteConnectOptions::new()
            .filename(":memory:")
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .expect("connect test database");
        startup::migrations::run(&pool)
            .await
            .expect("run migrations");
        startup::bootstrap::ensure_app_settings(&pool)
            .await
            .expect("ensure app settings");
        pool
    }

    async fn seed_user(pool: &SqlitePool, user_id: &str) {
        sqlx::query(
            r#"
            INSERT INTO users (
                id, username, password_hash, role, is_disabled, created_at, updated_at
            ) VALUES (?, ?, 'hash', 'user', 0, 1, 1)
            "#,
        )
        .bind(user_id)
        .bind(user_id)
        .execute(pool)
        .await
        .expect("insert test user");
    }

    async fn configure_embeddings(pool: &SqlitePool, base_url: &str, model: &str) {
        sqlx::query(
            r#"
            INSERT INTO ollama_backends (
                id, name, base_url, is_enabled, created_at, updated_at
            ) VALUES ('backend', 'Fake embeddings', ?, 1, 1, 1)
            ON CONFLICT(id) DO UPDATE SET base_url = excluded.base_url, is_enabled = 1
            "#,
        )
        .bind(base_url)
        .execute(pool)
        .await
        .expect("configure embedding backend");
        sqlx::query(
            r#"
            UPDATE app_settings
            SET notes_semantic_search_enabled = 1,
                notes_embedding_backend_id = 'backend',
                notes_embedding_model = ?
            WHERE id = 1
            "#,
        )
        .bind(model)
        .execute(pool)
        .await
        .expect("enable semantic search");
    }

    fn memory_request(content: &str, scope: MemoryModelScope) -> CreateMemoryRequest {
        CreateMemoryRequest {
            content: content.to_string(),
            model_scope: Some(scope),
        }
    }

    #[derive(Deserialize)]
    struct FakeEmbedRequest {
        input: Vec<String>,
    }

    async fn fake_embed(Json(request): Json<FakeEmbedRequest>) -> Json<serde_json::Value> {
        let embeddings = request
            .input
            .iter()
            .map(|text| {
                let text = text.to_ascii_lowercase();
                if text.contains("beans") || text.contains("favorite food") {
                    vec![1.0, 0.0, 0.0]
                } else if text.contains("orange") || text.contains("favorite color") {
                    vec![0.0, 1.0, 0.0]
                } else {
                    vec![0.0, 0.0, 1.0]
                }
            })
            .collect::<Vec<_>>();
        Json(json!({ "model": "fake-embed", "embeddings": embeddings }))
    }

    async fn fake_embedding_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fake embedding server");
        let address = listener.local_addr().expect("read fake server address");
        tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route("/api/embed", post(fake_embed)),
            )
            .await
            .expect("serve fake embeddings");
        });
        format!("http://{address}")
    }

    async fn process_all(retrieval: &MemoryRetrieval, client: &reqwest::Client) {
        while matches!(
            retrieval
                .process_next_job(client)
                .await
                .expect("process memory embedding"),
            JobResult::Indexed
        ) {}
    }

    #[tokio::test]
    async fn lifecycle_updates_persisted_vectors_and_enforces_scope() {
        let pool = test_pool().await;
        seed_user(&pool, "owner").await;
        seed_user(&pool, "other").await;
        let base_url = fake_embedding_server().await;
        configure_embeddings(&pool, &base_url, "fake-embed").await;
        let actor = memory_service::MemoryMutationActor::human("owner");
        let created = memory_service::create_memory(
            &pool,
            "owner",
            memory_request("The user really loves beans.", MemoryModelScope::default()),
            &actor,
        )
        .await
        .expect("create memory");
        let retrieval = MemoryRetrieval::new(pool.clone());
        let client = reqwest::Client::new();
        process_all(&retrieval, &client).await;

        let cold_retrieval = MemoryRetrieval::new(pool.clone());
        let matches = cold_retrieval
            .hybrid_search(&client, "owner", "base:test:model-a", "favorite food", 5)
            .await
            .expect("search persisted vectors with a cold cache");
        assert_eq!(
            matches.first().map(|memory| memory.id.as_str()),
            Some(created.id.as_str())
        );
        assert!(
            cold_retrieval
                .hybrid_search(&client, "other", "base:test:model-a", "favorite food", 5)
                .await
                .expect("search another user's memories")
                .is_empty()
        );

        let updated = memory_service::update_memory(
            &pool,
            "owner",
            &created.id,
            UpdateMemoryRequest {
                expected_version: 1,
                content: Some("The user's favorite color is orange.".to_string()),
                model_scope: Some(MemoryModelScope {
                    all_models: false,
                    model_keys: vec!["base:test:model-a".to_string()],
                }),
            },
            &actor,
        )
        .await
        .expect("update memory");
        process_all(&retrieval, &client).await;
        let stored_version: String = sqlx::query_scalar(
            "SELECT memory_version_id FROM memory_embeddings WHERE memory_id = ?",
        )
        .bind(&created.id)
        .fetch_one(&pool)
        .await
        .expect("read stored memory version");
        assert_eq!(stored_version, updated.current_version.id);
        assert_eq!(
            retrieval
                .hybrid_search(&client, "owner", "base:test:model-a", "favorite color", 5)
                .await
                .expect("search updated memory")
                .first()
                .map(|memory| memory.id.as_str()),
            Some(created.id.as_str())
        );
        assert!(
            retrieval
                .hybrid_search(&client, "owner", "base:test:model-b", "favorite color", 5)
                .await
                .expect("search outside model scope")
                .is_empty()
        );

        memory_service::forget_memory(&pool, "owner", &created.id, 2, &actor)
            .await
            .expect("forget memory");
        let indexed: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?")
                .bind(&created.id)
                .fetch_one(&pool)
                .await
                .expect("count forgotten embeddings");
        assert_eq!(indexed, 0);
        assert!(
            retrieval
                .hybrid_search(&client, "owner", "base:test:model-a", "favorite color", 5)
                .await
                .expect("search forgotten memories")
                .is_empty()
        );

        memory_service::restore_memory(&pool, "owner", &created.id, 2)
            .await
            .expect("restore memory");
        process_all(&retrieval, &client).await;
        assert_eq!(
            retrieval
                .hybrid_search(&client, "owner", "base:test:model-a", "favorite color", 5)
                .await
                .expect("search restored memory")
                .first()
                .map(|memory| memory.id.as_str()),
            Some(created.id.as_str())
        );
    }

    #[tokio::test]
    async fn failed_jobs_retry_and_model_changes_rebuild_vectors() {
        let pool = test_pool().await;
        seed_user(&pool, "owner").await;
        configure_embeddings(&pool, "http://127.0.0.1:9", "fake-embed").await;
        let actor = memory_service::MemoryMutationActor::human("owner");
        let created = memory_service::create_memory(
            &pool,
            "owner",
            memory_request("The user really loves beans.", MemoryModelScope::default()),
            &actor,
        )
        .await
        .expect("create memory");
        let retrieval = MemoryRetrieval::new(pool.clone());
        let client = reqwest::Client::new();
        assert!(retrieval.process_next_job(&client).await.is_err());
        let (indexed, pending, last_error) = index_status(&pool).await.expect("read failed status");
        assert_eq!((indexed, pending), (0, 1));
        assert!(last_error.is_some_and(|error| error.contains("embedding request failed")));

        let base_url = fake_embedding_server().await;
        configure_embeddings(&pool, &base_url, "fake-embed").await;
        sqlx::query("UPDATE memory_embedding_jobs SET last_attempt_at = NULL WHERE memory_id = ?")
            .bind(&created.id)
            .execute(&pool)
            .await
            .expect("make failed job immediately retryable");
        process_all(&retrieval, &client).await;
        assert_eq!(
            index_status(&pool).await.expect("read recovered status"),
            (1, 0, None)
        );

        configure_embeddings(&pool, &base_url, "fake-embed-v2").await;
        retrieval
            .rebuild_all()
            .await
            .expect("rebuild changed model");
        assert_eq!(index_status(&pool).await.expect("read rebuild status").0, 0);
        process_all(&retrieval, &client).await;
        let stored_model: String =
            sqlx::query_scalar("SELECT embedding_model FROM memory_embeddings WHERE memory_id = ?")
                .bind(&created.id)
                .fetch_one(&pool)
                .await
                .expect("read rebuilt embedding model");
        assert_eq!(stored_model, "fake-embed-v2");
    }
}
