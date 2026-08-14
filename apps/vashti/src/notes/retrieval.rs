use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use sha2::{Digest, Sha256};
use sqlx::{Row, SqlitePool};
use tokio::sync::{Mutex, Notify};
use uuid::Uuid;

use crate::{
    auth::service::unix_timestamp,
    notes::{models::NoteSummaryResponse, service as notes_service},
    ollama,
};

const CHUNKER_VERSION: i64 = 1;
const TARGET_CHUNK_CHARS: usize = 1_800;
const MAX_CHUNK_CHARS: usize = 2_400;
const CHUNK_OVERLAP_CHARS: usize = 160;
const MAX_INDEXED_NOTE_BYTES: usize = 256 * 1024;
const EMBED_BATCH_SIZE: usize = 24;
const MAX_CACHE_USERS: usize = 8;
const MAX_CACHE_BYTES: usize = 64 * 1024 * 1024;
const MAX_USER_CACHE_BYTES: usize = 16 * 1024 * 1024;
const FUSION_K: f64 = 60.0;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticConfig {
    pub enabled: bool,
    pub backend_id: Option<String>,
    pub model: Option<String>,
}

#[derive(Clone, Debug)]
struct ActiveSemanticConfig {
    backend_id: String,
    base_url: String,
    model: String,
}

#[derive(Clone, Debug)]
struct EmbeddingJob {
    note_id: String,
    user_id: String,
    note_version_id: String,
    title: String,
    content: String,
}

#[derive(Clone, Debug)]
struct IndexedChunk {
    text: String,
    hash: String,
    vector: Vec<f32>,
}

#[derive(Clone, Debug)]
struct CachedChunk {
    note_id: String,
    text: String,
    vector: Vec<f32>,
}

#[derive(Debug)]
struct CachedUserVectors {
    revision: i64,
    backend_id: String,
    model: String,
    chunks: Vec<CachedChunk>,
    bytes: usize,
    last_used: u64,
}

#[derive(Debug, Default)]
struct EmbeddingCache {
    users: HashMap<String, CachedUserVectors>,
    total_bytes: usize,
    clock: u64,
}

#[derive(Clone, Debug)]
struct SemanticMatch {
    note_id: String,
    excerpt: String,
    score: f32,
}

#[derive(Debug)]
enum JobResult {
    Indexed,
    NoWork,
}

#[derive(Debug)]
pub struct NoteRetrieval {
    pool: SqlitePool,
    cache: Mutex<EmbeddingCache>,
    notify: Notify,
}

impl NoteRetrieval {
    pub fn new(pool: SqlitePool) -> Self {
        Self {
            pool,
            cache: Mutex::new(EmbeddingCache::default()),
            notify: Notify::new(),
        }
    }

    pub fn wake(&self) {
        self.notify.notify_one();
    }

    pub async fn run(self: Arc<Self>, client: reqwest::Client) {
        loop {
            if let Err(error) = self.repair_index().await {
                tracing::warn!(%error, "failed to repair the note embedding queue");
            }

            loop {
                match self.process_next_job(&client).await {
                    Ok(JobResult::Indexed) => continue,
                    Ok(JobResult::NoWork) => break,
                    Err(error) => {
                        tracing::warn!(%error, "failed to index a note for semantic search");
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
        sqlx::query("DELETE FROM note_search_chunks")
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM note_embedding_jobs")
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            r#"
            INSERT INTO note_embedding_jobs (
                note_id, user_id, note_version_id, requested_at, attempts,
                last_attempt_at, last_error
            )
            SELECT id, user_id, current_version_id, updated_at, 0, NULL, NULL
            FROM notes
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
            INSERT INTO note_embedding_revisions (user_id, revision)
            SELECT DISTINCT user_id, 1 FROM notes
            WHERE 1 = 1
            ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1
            "#,
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query("UPDATE app_settings SET notes_embedding_last_error = NULL WHERE id = 1")
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
    ) -> Result<Vec<NoteSummaryResponse>, crate::error::ApiError> {
        let limit = limit.clamp(1, 10) as usize;
        let keyword =
            notes_service::search_notes_for_model(&self.pool, user_id, model_key, query, 10)
                .await?;

        let semantic = match self
            .semantic_search(client, user_id, model_key, query, 10)
            .await
        {
            Ok(matches) => matches,
            Err(error) => {
                tracing::warn!(%error, "semantic note search unavailable; using keyword results");
                return Ok(keyword.into_iter().take(limit).collect());
            }
        };
        if semantic.is_empty() {
            return Ok(keyword.into_iter().take(limit).collect());
        }

        let mut scores = HashMap::<String, f64>::new();
        let mut keyword_notes = HashMap::new();
        for (index, note) in keyword.into_iter().enumerate() {
            *scores.entry(note.id.clone()).or_default() += 1.15 / (FUSION_K + index as f64 + 1.0);
            keyword_notes.insert(note.id.clone(), note);
        }
        let mut semantic_excerpts = HashMap::new();
        for (index, note) in semantic.into_iter().enumerate() {
            *scores.entry(note.note_id.clone()).or_default() +=
                1.0 / (FUSION_K + index as f64 + 1.0);
            semantic_excerpts
                .entry(note.note_id)
                .or_insert(note.excerpt);
        }

        let mut ranked = scores.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        let mut results = Vec::with_capacity(limit);
        for (note_id, _) in ranked {
            if let Some(note) = keyword_notes.remove(&note_id) {
                results.push(note);
            } else if let Ok(note) =
                notes_service::get_note_for_model(&self.pool, user_id, &note_id, model_key).await
            {
                results.push(NoteSummaryResponse {
                    id: note.id,
                    title: note.current_version.title,
                    excerpt: semantic_excerpts.remove(&note_id).unwrap_or_default(),
                    current_version_id: note.current_version.id,
                    current_version_number: note.current_version.version_number,
                    tags: note.tags,
                    is_pinned: note.is_pinned,
                    ai_access: note.ai_access,
                    model_scope: note.model_scope,
                    deleted_at: note.deleted_at,
                    created_at: note.created_at,
                    updated_at: note.updated_at,
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
        let Some(config) = self.active_config().await? else {
            return Ok(Vec::new());
        };
        let allowed_note_ids = self.allowed_note_ids(user_id, model_key).await?;
        if allowed_note_ids.is_empty() {
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
        self.ensure_user_cache(user_id, revision, &config).await?;
        let mut cache = self.cache.lock().await;
        cache.clock = cache.clock.wrapping_add(1);
        let clock = cache.clock;
        let Some(entry) = cache.users.get_mut(user_id) else {
            return Ok(Vec::new());
        };
        entry.last_used = clock;
        let mut by_note = HashMap::<String, SemanticMatch>::new();
        for chunk in &entry.chunks {
            if !allowed_note_ids.contains(&chunk.note_id) {
                continue;
            }
            let Some(score) = cosine_similarity(&query_vector, &chunk.vector) else {
                continue;
            };
            let candidate = SemanticMatch {
                note_id: chunk.note_id.clone(),
                excerpt: truncate_excerpt(&chunk.text, 420),
                score,
            };
            match by_note.get(&chunk.note_id) {
                Some(current) if current.score >= score => {}
                _ => {
                    by_note.insert(chunk.note_id.clone(), candidate);
                }
            }
        }
        let mut matches = by_note.into_values().collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        matches.truncate(limit);
        Ok(matches)
    }

    async fn ensure_user_cache(
        &self,
        user_id: &str,
        revision: i64,
        config: &ActiveSemanticConfig,
    ) -> Result<(), String> {
        {
            let cache = self.cache.lock().await;
            if cache.users.get(user_id).is_some_and(|entry| {
                entry.revision == revision
                    && entry.backend_id == config.backend_id
                    && entry.model == config.model
            }) {
                return Ok(());
            }
        }

        let rows = sqlx::query(
            r#"
            SELECT chunks.note_id, chunks.chunk_text, chunks.vector
            FROM note_search_chunks chunks
            JOIN notes n ON n.id = chunks.note_id
            WHERE chunks.user_id = ?
              AND chunks.note_version_id = n.current_version_id
              AND n.deleted_at IS NULL
              AND chunks.embedding_backend_id = ?
              AND chunks.embedding_model = ?
              AND chunks.chunker_version = ?
            ORDER BY n.updated_at DESC, chunks.chunk_index ASC
            "#,
        )
        .bind(user_id)
        .bind(&config.backend_id)
        .bind(&config.model)
        .bind(CHUNKER_VERSION)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("failed to load note vectors: {error}"))?;

        let mut chunks = Vec::new();
        let mut bytes = 0usize;
        for row in rows {
            let vector_bytes: Vec<u8> = row
                .try_get("vector")
                .map_err(|error| format!("invalid stored note vector: {error}"))?;
            let vector = vector_from_bytes(&vector_bytes)?;
            let text: String = row
                .try_get("chunk_text")
                .map_err(|error| format!("invalid stored note chunk: {error}"))?;
            let chunk_bytes = vector_bytes.len().saturating_add(text.len());
            if bytes.saturating_add(chunk_bytes) > MAX_USER_CACHE_BYTES {
                break;
            }
            bytes += chunk_bytes;
            chunks.push(CachedChunk {
                note_id: row
                    .try_get("note_id")
                    .map_err(|error| format!("invalid stored note ID: {error}"))?,
                text,
                vector,
            });
        }

        let mut cache = self.cache.lock().await;
        if let Some(previous) = cache.users.remove(user_id) {
            cache.total_bytes = cache.total_bytes.saturating_sub(previous.bytes);
        }
        cache.clock = cache.clock.wrapping_add(1);
        let last_used = cache.clock;
        cache.total_bytes = cache.total_bytes.saturating_add(bytes);
        cache.users.insert(
            user_id.to_string(),
            CachedUserVectors {
                revision,
                backend_id: config.backend_id.clone(),
                model: config.model.clone(),
                chunks,
                bytes,
                last_used,
            },
        );
        while cache.users.len() > MAX_CACHE_USERS || cache.total_bytes > MAX_CACHE_BYTES {
            let Some(oldest) = cache
                .users
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(user_id, _)| user_id.clone())
            else {
                break;
            };
            if let Some(entry) = cache.users.remove(&oldest) {
                cache.total_bytes = cache.total_bytes.saturating_sub(entry.bytes);
            }
        }
        Ok(())
    }

    async fn allowed_note_ids(
        &self,
        user_id: &str,
        model_key: &str,
    ) -> Result<HashSet<String>, String> {
        sqlx::query_scalar(
            r#"
            SELECT n.id
            FROM notes n
            JOIN user_note_settings settings ON settings.user_id = n.user_id
            WHERE n.user_id = ?
              AND n.deleted_at IS NULL
              AND settings.allow_model_read = 1
              AND n.ai_access IN ('read', 'edit', 'manage')
              AND (
                  n.all_models = 1
                  OR EXISTS (
                      SELECT 1 FROM note_model_scopes scopes
                      WHERE scopes.note_id = n.id AND scopes.model_key = ?
                  )
              )
            "#,
        )
        .bind(user_id)
        .bind(model_key)
        .fetch_all(&self.pool)
        .await
        .map(|ids| ids.into_iter().collect())
        .map_err(|error| format!("failed to apply note permissions: {error}"))
    }

    async fn user_revision(&self, user_id: &str) -> Result<i64, String> {
        sqlx::query_scalar("SELECT revision FROM note_embedding_revisions WHERE user_id = ?")
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map(|revision| revision.unwrap_or(0))
            .map_err(|error| format!("failed to read note index revision: {error}"))
    }

    async fn process_next_job(&self, client: &reqwest::Client) -> Result<JobResult, String> {
        let Some(config) = self.active_config().await? else {
            return Ok(JobResult::NoWork);
        };
        let now = unix_timestamp();
        let row = sqlx::query(
            r#"
            SELECT jobs.note_id, jobs.user_id, jobs.note_version_id,
                   versions.title, versions.content
            FROM note_embedding_jobs jobs
            JOIN notes n ON n.id = jobs.note_id
            JOIN note_versions versions ON versions.id = jobs.note_version_id
            WHERE n.deleted_at IS NULL
              AND n.current_version_id = jobs.note_version_id
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
        .map_err(|error| format!("failed to read note embedding queue: {error}"))?;
        let Some(row) = row else {
            return Ok(JobResult::NoWork);
        };
        let job = EmbeddingJob {
            note_id: row.try_get("note_id").map_err(db_value_error)?,
            user_id: row.try_get("user_id").map_err(db_value_error)?,
            note_version_id: row.try_get("note_version_id").map_err(db_value_error)?,
            title: row.try_get("title").map_err(db_value_error)?,
            content: row.try_get("content").map_err(db_value_error)?,
        };
        let texts = chunk_markdown(&job.title, &job.content);
        let mut indexed = Vec::with_capacity(texts.len());
        let mut dimensions = None;

        for batch in texts.chunks(EMBED_BATCH_SIZE) {
            let response =
                match ollama::client::embed(client, &config.base_url, &config.model, batch).await {
                    Ok(response) => response,
                    Err(error) => {
                        let message = format!("embedding request failed: {error}");
                        self.record_failure(&job.note_id, &message).await;
                        return Err(message);
                    }
                };
            if response.embeddings.len() != batch.len() {
                let message = format!(
                    "embedding backend returned {} vectors for {} chunks",
                    response.embeddings.len(),
                    batch.len()
                );
                self.record_failure(&job.note_id, &message).await;
                return Err(message);
            }
            for (text, vector) in batch.iter().zip(response.embeddings) {
                if let Err(message) = validate_vector(&vector, dimensions) {
                    self.record_failure(&job.note_id, &message).await;
                    return Err(message);
                }
                dimensions = Some(vector.len());
                indexed.push(IndexedChunk {
                    text: text.clone(),
                    hash: hex_sha256(text.as_bytes()),
                    vector,
                });
            }
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("failed to save note embeddings: {error}"))?;
        let still_current: i64 = sqlx::query_scalar(
            r#"
            SELECT EXISTS(
                SELECT 1
                FROM notes n
                JOIN app_settings settings ON settings.id = 1
                WHERE n.id = ?
                  AND n.current_version_id = ?
                  AND n.deleted_at IS NULL
                  AND settings.notes_semantic_search_enabled = 1
                  AND settings.notes_embedding_backend_id = ?
                  AND settings.notes_embedding_model = ?
            )
            "#,
        )
        .bind(&job.note_id)
        .bind(&job.note_version_id)
        .bind(&config.backend_id)
        .bind(&config.model)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("failed to verify note embedding source: {error}"))?;
        if still_current == 0 {
            tx.rollback().await.ok();
            return Ok(JobResult::Indexed);
        }

        sqlx::query("DELETE FROM note_search_chunks WHERE note_id = ?")
            .bind(&job.note_id)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("failed to replace note embeddings: {error}"))?;
        for (index, chunk) in indexed.iter().enumerate() {
            sqlx::query(
                r#"
                INSERT INTO note_search_chunks (
                    id, user_id, note_id, note_version_id, chunk_index, chunk_text,
                    chunk_text_hash, embedding_backend_id, embedding_model,
                    embedding_dimensions, chunker_version, vector, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#,
            )
            .bind(Uuid::new_v4().to_string())
            .bind(&job.user_id)
            .bind(&job.note_id)
            .bind(&job.note_version_id)
            .bind(index as i64)
            .bind(&chunk.text)
            .bind(&chunk.hash)
            .bind(&config.backend_id)
            .bind(&config.model)
            .bind(chunk.vector.len() as i64)
            .bind(CHUNKER_VERSION)
            .bind(vector_to_bytes(&chunk.vector))
            .bind(now)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("failed to insert note embedding: {error}"))?;
        }
        sqlx::query("DELETE FROM note_embedding_jobs WHERE note_id = ? AND note_version_id = ?")
            .bind(&job.note_id)
            .bind(&job.note_version_id)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("failed to finish note embedding job: {error}"))?;
        sqlx::query(
            r#"
            INSERT INTO note_embedding_revisions (user_id, revision)
            VALUES (?, 1)
            ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1
            "#,
        )
        .bind(&job.user_id)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("failed to publish note index revision: {error}"))?;
        sqlx::query("UPDATE app_settings SET notes_embedding_last_error = NULL WHERE id = 1")
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("failed to clear note index status: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("failed to commit note embeddings: {error}"))?;
        Ok(JobResult::Indexed)
    }

    async fn repair_index(&self) -> Result<(), sqlx::Error> {
        let Some(config) = self.active_config().await.map_err(sqlx::Error::Protocol)? else {
            return Ok(());
        };
        sqlx::query(
            r#"
            INSERT INTO note_embedding_jobs (
                note_id, user_id, note_version_id, requested_at, attempts,
                last_attempt_at, last_error
            )
            SELECT n.id, n.user_id, n.current_version_id, n.updated_at, 0, NULL, NULL
            FROM notes n
            WHERE n.deleted_at IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM note_search_chunks chunks
                  WHERE chunks.note_id = n.id
                    AND chunks.note_version_id = n.current_version_id
                    AND chunks.embedding_backend_id = ?
                    AND chunks.embedding_model = ?
                    AND chunks.chunker_version = ?
              )
            ON CONFLICT(note_id) DO UPDATE SET
                user_id = excluded.user_id,
                note_version_id = excluded.note_version_id,
                requested_at = excluded.requested_at
            "#,
        )
        .bind(config.backend_id)
        .bind(config.model)
        .bind(CHUNKER_VERSION)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn active_config(&self) -> Result<Option<ActiveSemanticConfig>, String> {
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
        .map_err(|error| format!("failed to read semantic note settings: {error}"))?;
        let enabled = row
            .try_get::<i64, _>("notes_semantic_search_enabled")
            .map_err(db_value_error)?
            != 0;
        if !enabled {
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
                Ok(Some(ActiveSemanticConfig {
                    backend_id,
                    base_url,
                    model,
                }))
            }
            _ => Err("semantic note search has no enabled embedding backend and model".to_string()),
        }
    }

    async fn record_failure(&self, note_id: &str, error: &str) {
        let error = truncate_excerpt(error, 500);
        let now = unix_timestamp();
        if let Err(db_error) = sqlx::query(
            r#"
            UPDATE note_embedding_jobs
            SET attempts = attempts + 1, last_attempt_at = ?, last_error = ?
            WHERE note_id = ?
            "#,
        )
        .bind(now)
        .bind(&error)
        .bind(note_id)
        .execute(&self.pool)
        .await
        {
            tracing::warn!(?db_error, "failed to record note embedding job error");
        }
        if let Err(db_error) =
            sqlx::query("UPDATE app_settings SET notes_embedding_last_error = ? WHERE id = 1")
                .bind(error)
                .execute(&self.pool)
                .await
        {
            tracing::warn!(?db_error, "failed to record note embedding status");
        }
    }
}

pub async fn semantic_config(pool: &SqlitePool) -> Result<SemanticConfig, sqlx::Error> {
    let row = sqlx::query(
        r#"
        SELECT notes_semantic_search_enabled, notes_embedding_backend_id,
               notes_embedding_model
        FROM app_settings
        WHERE id = 1
        "#,
    )
    .fetch_one(pool)
    .await?;
    Ok(SemanticConfig {
        enabled: row.try_get::<i64, _>("notes_semantic_search_enabled")? != 0,
        backend_id: row.try_get("notes_embedding_backend_id")?,
        model: row.try_get("notes_embedding_model")?,
    })
}

pub async fn index_status(pool: &SqlitePool) -> Result<(i64, i64, Option<String>), sqlx::Error> {
    let indexed = sqlx::query_scalar("SELECT COUNT(*) FROM note_search_chunks")
        .fetch_one(pool)
        .await?;
    let pending = sqlx::query_scalar("SELECT COUNT(*) FROM note_embedding_jobs")
        .fetch_one(pool)
        .await?;
    let last_error =
        sqlx::query_scalar("SELECT notes_embedding_last_error FROM app_settings WHERE id = 1")
            .fetch_one(pool)
            .await?;
    Ok((indexed, pending, last_error))
}

fn chunk_markdown(title: &str, content: &str) -> Vec<String> {
    let content = truncate_bytes(content, MAX_INDEXED_NOTE_BYTES);
    let blocks = markdown_blocks(content);
    let mut chunks = Vec::new();
    let mut current = String::new();
    for block in blocks {
        if block.chars().count() > MAX_CHUNK_CHARS {
            if !current.trim().is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            chunks.extend(split_large_block(&block));
            continue;
        }
        let additional = block.chars().count() + usize::from(!current.is_empty()) * 2;
        if !current.is_empty()
            && current.chars().count().saturating_add(additional) > TARGET_CHUNK_CHARS
        {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(&block);
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    let heading = format!("# {}", title.trim());
    chunks
        .into_iter()
        .map(|chunk| {
            if chunk.trim().is_empty() {
                heading.clone()
            } else {
                format!("{heading}\n\n{}", chunk.trim())
            }
        })
        .collect()
}

fn markdown_blocks(content: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut in_fence = false;
    for line in content.lines() {
        let trimmed = line.trim_start();
        let is_fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");
        let is_heading = !in_fence && trimmed.starts_with('#');
        if is_heading && !current.trim().is_empty() {
            blocks.push(std::mem::take(&mut current));
        }
        if !in_fence && line.trim().is_empty() {
            if !current.trim().is_empty() {
                blocks.push(std::mem::take(&mut current));
            }
            continue;
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
        if is_fence {
            in_fence = !in_fence;
        }
        if is_heading && !current.trim().is_empty() {
            blocks.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        blocks.push(current);
    }
    blocks
}

fn split_large_block(block: &str) -> Vec<String> {
    let chars = block.chars().collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let end = (start + MAX_CHUNK_CHARS).min(chars.len());
        chunks.push(chars[start..end].iter().collect());
        if end == chars.len() {
            break;
        }
        start = end.saturating_sub(CHUNK_OVERLAP_CHARS);
    }
    chunks
}

fn truncate_bytes(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn truncate_excerpt(value: &str, max_chars: usize) -> String {
    let mut result = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        result.push('…');
    }
    result
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

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (&left, &right) in left.iter().zip(right) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return None;
    }
    Some((dot / (left_norm.sqrt() * right_norm.sqrt())) as f32)
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
        return Err("stored note embedding has an invalid byte length".to_string());
    }
    let vector = bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    validate_vector(&vector, None)?;
    Ok(vector)
}

fn hex_sha256(value: &[u8]) -> String {
    let digest = Sha256::digest(value);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn db_value_error(error: sqlx::Error) -> String {
    format!("invalid note index database value: {error}")
}

#[cfg(test)]
mod tests {
    use axum::{Json, Router, routing::post};
    use serde::Deserialize;
    use serde_json::json;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    use super::*;
    use crate::{
        auth::service::register_user,
        notes::models::{
            CreateNoteRequest, NoteAiAccess, NoteModelScope, UpdateNoteRequest,
            UpdateNoteSettingsRequest,
        },
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

    async fn create_test_user(pool: &SqlitePool) -> String {
        register_user(
            pool,
            "retrieval-user".to_string(),
            None,
            "secret-pass".to_string(),
        )
        .await
        .expect("register test user")
        .user
        .id
    }

    fn note_request(title: &str, content: &str) -> CreateNoteRequest {
        CreateNoteRequest {
            title: title.to_string(),
            content: content.to_string(),
            tags: Vec::new(),
            is_pinned: false,
            ai_access: Some(NoteAiAccess::Read),
            model_scope: Some(NoteModelScope::default()),
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
                if text.contains("rust") || text.contains("compiler") || text.contains("ownership")
                {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
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

    #[test]
    fn markdown_chunking_preserves_fenced_code_and_title() {
        let code = "```rust\nfn main() {\n    println!(\"hello\");\n}\n```";
        let content = format!("## Setup\n\nIntro paragraph.\n\n{code}\n\n## End\n\nDone.");
        let chunks = chunk_markdown("Project", &content);

        assert!(chunks.iter().all(|chunk| chunk.starts_with("# Project")));
        assert!(chunks.iter().any(|chunk| chunk.contains(code)));
    }

    #[test]
    fn markdown_chunking_caps_large_blocks_with_overlap() {
        let content = "x".repeat(MAX_CHUNK_CHARS * 3);
        let chunks = chunk_markdown("Large", &content);

        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|chunk| {
            chunk.chars().count() <= MAX_CHUNK_CHARS + "# Large\n\n".chars().count()
        }));
    }

    #[test]
    fn vectors_round_trip_and_cosine_is_exact() {
        let vector = vec![0.25, -0.5, 1.0];
        assert_eq!(
            vector_from_bytes(&vector_to_bytes(&vector)).unwrap(),
            vector
        );
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), Some(1.0));
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), Some(0.0));
    }

    #[test]
    fn invalid_vectors_are_rejected() {
        assert!(validate_vector(&[], None).is_err());
        assert!(validate_vector(&[f32::NAN], None).is_err());
        assert!(validate_vector(&[1.0], Some(2)).is_err());
        assert!(vector_from_bytes(&[1, 2, 3]).is_err());
    }

    #[tokio::test]
    async fn note_lifecycle_keeps_embedding_queue_current() {
        let pool = test_pool().await;
        let user_id = create_test_user(&pool).await;
        let actor = notes_service::NoteMutationActor::human(&user_id);
        let created = notes_service::create_note(
            &pool,
            &user_id,
            note_request("Project", "Initial content"),
            &actor,
        )
        .await
        .expect("create note");

        let queued_version: String =
            sqlx::query_scalar("SELECT note_version_id FROM note_embedding_jobs WHERE note_id = ?")
                .bind(&created.id)
                .fetch_one(&pool)
                .await
                .expect("read queued version");
        assert_eq!(queued_version, created.current_version.id);

        let updated = notes_service::update_note(
            &pool,
            &user_id,
            &created.id,
            UpdateNoteRequest {
                expected_version: 1,
                expected_version_id: None,
                edit_session_version_id: None,
                title: None,
                content: Some("Updated content".to_string()),
                tags: None,
                is_pinned: None,
                ai_access: None,
                model_scope: None,
            },
            &actor,
        )
        .await
        .expect("update note");
        let queued_version: String =
            sqlx::query_scalar("SELECT note_version_id FROM note_embedding_jobs WHERE note_id = ?")
                .bind(&created.id)
                .fetch_one(&pool)
                .await
                .expect("read updated queued version");
        assert_eq!(queued_version, updated.current_version.id);

        notes_service::trash_note(&pool, &user_id, &created.id, 2, &actor)
            .await
            .expect("trash note");
        let queued: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM note_embedding_jobs WHERE note_id = ?")
                .bind(&created.id)
                .fetch_one(&pool)
                .await
                .expect("count queued notes");
        assert_eq!(queued, 0);

        notes_service::restore_note(&pool, &user_id, &created.id, 2)
            .await
            .expect("restore note");
        let queued: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM note_embedding_jobs WHERE note_id = ?")
                .bind(&created.id)
                .fetch_one(&pool)
                .await
                .expect("count restored queue");
        assert_eq!(queued, 1);
    }

    #[tokio::test]
    async fn hybrid_search_uses_semantics_and_falls_back_to_keywords() {
        let pool = test_pool().await;
        let user_id = create_test_user(&pool).await;
        let actor = notes_service::NoteMutationActor::human(&user_id);
        notes_service::update_note_settings(
            &pool,
            &user_id,
            UpdateNoteSettingsRequest {
                allow_model_read: true,
                allow_model_create: false,
                allow_model_edit: false,
                allow_model_trash: false,
                default_ai_access: NoteAiAccess::Read,
                default_model_scope: NoteModelScope::default(),
            },
        )
        .await
        .expect("enable note reading");
        let rust_note = notes_service::create_note(
            &pool,
            &user_id,
            note_request("Rust guide", "Borrowed values are checked by the compiler."),
            &actor,
        )
        .await
        .expect("create Rust note");
        notes_service::create_note(
            &pool,
            &user_id,
            note_request("Beach plan", "Pack a towel and sunscreen for Saturday."),
            &actor,
        )
        .await
        .expect("create beach note");

        let backend_id = Uuid::new_v4().to_string();
        let base_url = fake_embedding_server().await;
        let now = unix_timestamp();
        sqlx::query(
            "INSERT INTO ollama_backends (id, name, base_url, created_at, updated_at) VALUES (?, 'Fake embeddings', ?, ?, ?)",
        )
        .bind(&backend_id)
        .bind(&base_url)
        .bind(now)
        .bind(now)
        .execute(&pool)
        .await
        .expect("create embedding backend");
        sqlx::query(
            "INSERT INTO model_availability (backend_id, model_name, is_enabled, created_at, updated_at) VALUES (?, 'fake-embed', 1, ?, ?)",
        )
        .bind(&backend_id)
        .bind(now)
        .bind(now)
        .execute(&pool)
        .await
        .expect("enable embedding model");
        sqlx::query(
            "UPDATE app_settings SET notes_semantic_search_enabled = 1, notes_embedding_backend_id = ?, notes_embedding_model = 'fake-embed' WHERE id = 1",
        )
        .bind(&backend_id)
        .execute(&pool)
        .await
        .expect("enable semantic search");

        let retrieval = NoteRetrieval::new(pool.clone());
        let client = reqwest::Client::new();
        while matches!(
            retrieval
                .process_next_job(&client)
                .await
                .expect("index note"),
            JobResult::Indexed
        ) {}

        let semantic = retrieval
            .hybrid_search(
                &client,
                &user_id,
                "backend:test-model",
                "ownership rules",
                5,
            )
            .await
            .expect("semantic search");
        assert_eq!(
            semantic.first().map(|note| note.id.as_str()),
            Some(rust_note.id.as_str())
        );

        sqlx::query("UPDATE app_settings SET notes_semantic_search_enabled = 0 WHERE id = 1")
            .execute(&pool)
            .await
            .expect("disable semantic search");
        let keyword = retrieval
            .hybrid_search(&client, &user_id, "backend:test-model", "Saturday", 5)
            .await
            .expect("keyword fallback");
        assert_eq!(
            keyword.first().map(|note| note.title.as_str()),
            Some("Beach plan")
        );
    }
}
