use std::{
    collections::HashMap,
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};

use serde::Serialize;
use sqlx::{Row, SqlitePool};
use tokio::sync::{Mutex, Notify};

use crate::{
    auth::service::unix_timestamp,
    error::ApiError,
    ollama,
    vector_index::{VectorIndex, normalize},
};

const MIN_CACHE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CACHE_BYTES: usize = 512 * 1024 * 1024;
const FALLBACK_CACHE_BYTES: usize = 128 * 1024 * 1024;
const FUSION_K: f64 = 60.0;
const MAX_EMBEDDING_CHARS: usize = 16_000;
const MAX_RESULT_EXCERPT_CHARS: usize = 520;
const MAX_CONTEXT_MESSAGE_CHARS: usize = 6_000;
const MAX_CONTEXT_MESSAGES: i64 = 12;

#[derive(Clone, Debug)]
struct ActiveConfig {
    backend_id: String,
    base_url: String,
    model: String,
}

#[derive(Clone, Debug)]
struct EmbeddingJob {
    message_id: String,
    user_id: String,
    revision_id: String,
    chat_title: String,
    role: String,
    content: String,
}

#[derive(Clone, Debug)]
struct CachedConversation {
    message_id: String,
}

#[derive(Debug)]
struct CachedUserVectors {
    revision: i64,
    backend_id: String,
    model: String,
    index: VectorIndex<CachedConversation>,
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
    message_id: String,
    score: f32,
}

#[derive(Debug)]
enum JobResult {
    Indexed,
    NoWork,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConversationSearchResult {
    pub chat_id: String,
    pub message_id: String,
    pub chat_title: String,
    pub role: String,
    pub excerpt: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConversationContextMessage {
    pub message_id: String,
    pub role: String,
    pub content: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConversationContext {
    pub chat_id: String,
    pub chat_title: String,
    pub target_message_id: String,
    pub messages: Vec<ConversationContextMessage>,
}

#[derive(Debug)]
pub struct ConversationRetrieval {
    pool: SqlitePool,
    cache: Mutex<EmbeddingCache>,
    cache_budget_bytes: usize,
    notify: Notify,
}

impl ConversationRetrieval {
    pub fn new(pool: SqlitePool) -> Self {
        let cache_budget_bytes = adaptive_cache_bytes();
        tracing::info!(
            cache_budget_mib = cache_budget_bytes / (1024 * 1024),
            "configured conversation vector cache"
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
                tracing::warn!(%error, "failed to repair the conversation search index");
            }
            loop {
                match self.process_next_job(&client).await {
                    Ok(JobResult::Indexed) => continue,
                    Ok(JobResult::NoWork) => break,
                    Err(error) => {
                        tracing::warn!(%error, "failed to index a conversation message");
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
        self.sync_documents().await?;
        let mut tx = self.pool.begin().await?;
        sqlx::query("DELETE FROM conversation_embeddings")
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM conversation_embedding_jobs")
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            r#"
            INSERT INTO conversation_embedding_jobs (
                message_id, user_id, revision_id, requested_at,
                attempts, last_attempt_at, last_error
            )
            SELECT message_id, user_id, revision_id, source_updated_at, 0, NULL, NULL
            FROM conversation_search_documents
            WHERE EXISTS (
                SELECT 1 FROM app_settings
                WHERE id = 1 AND notes_semantic_search_enabled = 1
            )
            "#,
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            r#"
            INSERT INTO conversation_embedding_revisions (user_id, revision)
            SELECT DISTINCT user_id, 1 FROM conversation_search_documents
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
        query: &str,
        limit: i64,
    ) -> Result<Vec<ConversationSearchResult>, ApiError> {
        self.ensure_model_access(user_id).await?;
        self.sync_documents().await?;
        let limit = limit.clamp(1, 10) as usize;
        let keyword = self.keyword_search(user_id, query, 10).await?;
        let semantic = match self.semantic_search(client, user_id, query, 10).await {
            Ok(matches) => matches,
            Err(error) => {
                tracing::warn!(%error, "semantic conversation search unavailable; using keywords");
                return Ok(keyword.into_iter().take(limit).collect());
            }
        };
        if semantic.is_empty() {
            return Ok(keyword.into_iter().take(limit).collect());
        }

        let mut scores = HashMap::<String, f64>::new();
        let mut results_by_message = HashMap::new();
        for (index, result) in keyword.into_iter().enumerate() {
            *scores.entry(result.message_id.clone()).or_default() +=
                1.15 / (FUSION_K + index as f64 + 1.0);
            results_by_message.insert(result.message_id.clone(), result);
        }
        for (index, result) in semantic.into_iter().enumerate() {
            *scores.entry(result.message_id).or_default() += 1.0 / (FUSION_K + index as f64 + 1.0);
        }

        let mut ranked = scores.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        let mut results = Vec::with_capacity(limit);
        for (message_id, _) in ranked {
            if let Some(result) = results_by_message.remove(&message_id) {
                results.push(result);
            } else if let Some(result) = self.result_for_message(user_id, &message_id).await? {
                results.push(result);
            }
            if results.len() == limit {
                break;
            }
        }
        Ok(results)
    }

    pub async fn read_context(
        &self,
        user_id: &str,
        chat_id: &str,
        message_id: &str,
        limit: i64,
    ) -> Result<ConversationContext, ApiError> {
        self.ensure_model_access(user_id).await?;
        let target = sqlx::query(
            r#"
            SELECT documents.chat_title
            FROM conversation_search_documents documents
            WHERE documents.user_id = ?
              AND documents.chat_id = ?
              AND documents.message_id = ?
            "#,
        )
        .bind(user_id)
        .bind(chat_id)
        .bind(message_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| {
            ApiError::not_found(
                "conversation_message_not_found",
                "Searchable conversation message not found",
            )
        })?;
        let chat_title: String = target.try_get("chat_title")?;
        let rows = sqlx::query(
            r#"
            WITH RECURSIVE lineage (
                message_id, parent_message_id, role, status,
                active_revision_id, created_at, depth
            ) AS (
                SELECT id, parent_message_id, role, status,
                       active_revision_id, created_at, 0
                FROM chat_messages
                WHERE id = ? AND chat_id = ? AND is_deleted = 0
                UNION ALL
                SELECT parent.id, parent.parent_message_id, parent.role, parent.status,
                       parent.active_revision_id, parent.created_at, lineage.depth + 1
                FROM chat_messages parent
                JOIN lineage ON lineage.parent_message_id = parent.id
                WHERE parent.chat_id = ? AND parent.is_deleted = 0 AND lineage.depth < 63
            )
            SELECT lineage.message_id, lineage.role, revisions.content_text, lineage.created_at,
                   lineage.depth
            FROM lineage
            JOIN chat_message_revisions revisions ON revisions.id = lineage.active_revision_id
            WHERE lineage.role IN ('user', 'assistant')
              AND lineage.status = 'complete'
              AND trim(revisions.content_text) != ''
            ORDER BY lineage.depth ASC
            LIMIT ?
            "#,
        )
        .bind(message_id)
        .bind(chat_id)
        .bind(chat_id)
        .bind(limit.clamp(1, MAX_CONTEXT_MESSAGES))
        .fetch_all(&self.pool)
        .await?;
        let mut messages = rows
            .into_iter()
            .map(|row| {
                Ok(ConversationContextMessage {
                    message_id: row.try_get("message_id")?,
                    role: row.try_get("role")?,
                    content: truncate(
                        &row.try_get::<String, _>("content_text")?,
                        MAX_CONTEXT_MESSAGE_CHARS,
                    ),
                    created_at: row.try_get("created_at")?,
                })
            })
            .collect::<Result<Vec<_>, sqlx::Error>>()?;
        messages.reverse();
        Ok(ConversationContext {
            chat_id: chat_id.to_string(),
            chat_title,
            target_message_id: message_id.to_string(),
            messages,
        })
    }

    async fn ensure_model_access(&self, user_id: &str) -> Result<(), ApiError> {
        let allowed: bool = sqlx::query_scalar(
            r#"
            SELECT COALESCE((
                SELECT allow_model_chat_history
                FROM user_memory_settings
                WHERE user_id = ?
            ), 0)
            "#,
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await?;
        if allowed {
            Ok(())
        } else {
            Err(ApiError::forbidden(
                "conversation_search_disabled",
                "Past-chat access is disabled in personal settings",
            ))
        }
    }

    async fn keyword_search(
        &self,
        user_id: &str,
        query: &str,
        limit: i64,
    ) -> Result<Vec<ConversationSearchResult>, ApiError> {
        let query = build_fts_query(query);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let rows = sqlx::query(
            r#"
            SELECT documents.chat_id, documents.message_id, documents.chat_title,
                   documents.role,
                   snippet(conversation_search_fts, 1, '', '', ' ... ', 48) AS excerpt,
                   messages.created_at
            FROM conversation_search_fts
            JOIN conversation_search_documents documents
              ON documents.rowid = conversation_search_fts.rowid
            JOIN chat_messages messages ON messages.id = documents.message_id
            WHERE conversation_search_fts MATCH ?
              AND documents.user_id = ?
            ORDER BY bm25(conversation_search_fts, 1.8, 1.0) ASC,
                     documents.source_updated_at DESC
            LIMIT ?
            "#,
        )
        .bind(query)
        .bind(user_id)
        .bind(limit.clamp(1, 10))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(row_to_search_result)
            .collect::<Result<Vec<_>, sqlx::Error>>()?)
    }

    async fn result_for_message(
        &self,
        user_id: &str,
        message_id: &str,
    ) -> Result<Option<ConversationSearchResult>, ApiError> {
        let row = sqlx::query(
            r#"
            SELECT documents.chat_id, documents.message_id, documents.chat_title,
                   documents.role, substr(documents.content, 1, 520) AS excerpt,
                   messages.created_at
            FROM conversation_search_documents documents
            JOIN chat_messages messages ON messages.id = documents.message_id
            WHERE documents.user_id = ? AND documents.message_id = ?
            "#,
        )
        .bind(user_id)
        .bind(message_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(row_to_search_result).transpose()?)
    }

    async fn semantic_search(
        &self,
        client: &reqwest::Client,
        user_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SemanticMatch>, String> {
        let started = Instant::now();
        let Some(config) = self.active_config().await? else {
            return Ok(Vec::new());
        };
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
            .for_each_score(&query_vector, |_, conversation, score| {
                matches.push(SemanticMatch {
                    message_id: conversation.message_id.clone(),
                    score,
                });
            })?;
        matches.sort_unstable_by(|left, right| right.score.total_cmp(&left.score));
        matches.truncate(limit);
        tracing::debug!(
            user_id,
            indexed_messages = vectors.index.len(),
            result_count = matches.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "completed semantic conversation search"
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
            SELECT embeddings.message_id, embeddings.vector
            FROM conversation_embeddings embeddings
            JOIN conversation_search_documents documents
              ON documents.message_id = embeddings.message_id
            WHERE embeddings.user_id = ?
              AND embeddings.revision_id = documents.revision_id
              AND embeddings.embedding_backend_id = ?
              AND embeddings.embedding_model = ?
            ORDER BY documents.source_updated_at DESC
            "#,
        )
        .bind(user_id)
        .bind(&config.backend_id)
        .bind(&config.model)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("failed to load conversation vectors: {error}"))?;
        let mut entries = Vec::with_capacity(rows.len());
        let mut metadata_bytes = 0usize;
        for row in rows {
            let message_id: String = row.try_get("message_id").map_err(db_value_error)?;
            let vector_bytes: Vec<u8> = row.try_get("vector").map_err(db_value_error)?;
            metadata_bytes = metadata_bytes
                .saturating_add(message_id.capacity())
                .saturating_add(size_of::<CachedConversation>());
            entries.push((
                CachedConversation { message_id },
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

    async fn user_revision(&self, user_id: &str) -> Result<i64, String> {
        sqlx::query_scalar(
            "SELECT revision FROM conversation_embedding_revisions WHERE user_id = ?",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
        .map(|revision| revision.unwrap_or(0))
        .map_err(|error| format!("failed to read conversation index revision: {error}"))
    }

    async fn process_next_job(&self, client: &reqwest::Client) -> Result<JobResult, String> {
        let Some(config) = self.active_config().await? else {
            return Ok(JobResult::NoWork);
        };
        let now = unix_timestamp();
        let row = sqlx::query(
            r#"
            SELECT jobs.message_id, jobs.user_id, jobs.revision_id,
                   documents.chat_title, documents.role, documents.content
            FROM conversation_embedding_jobs jobs
            JOIN conversation_search_documents documents
              ON documents.message_id = jobs.message_id
             AND documents.revision_id = jobs.revision_id
            WHERE jobs.last_attempt_at IS NULL
               OR jobs.last_attempt_at <= ? - MIN(300, MAX(15, jobs.attempts * 15))
            ORDER BY jobs.requested_at ASC
            LIMIT 1
            "#,
        )
        .bind(now)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("failed to read conversation embedding queue: {error}"))?;
        let Some(row) = row else {
            return Ok(JobResult::NoWork);
        };
        let job = EmbeddingJob {
            message_id: row.try_get("message_id").map_err(db_value_error)?,
            user_id: row.try_get("user_id").map_err(db_value_error)?,
            revision_id: row.try_get("revision_id").map_err(db_value_error)?,
            chat_title: row.try_get("chat_title").map_err(db_value_error)?,
            role: row.try_get("role").map_err(db_value_error)?,
            content: row.try_get("content").map_err(db_value_error)?,
        };
        let input = embedding_input(&job.chat_title, &job.role, &job.content);
        let response = match ollama::client::embed(
            client,
            &config.base_url,
            &config.model,
            std::slice::from_ref(&input),
        )
        .await
        {
            Ok(response) => response,
            Err(error) => {
                let message = format!("embedding request failed: {error}");
                self.record_failure(&job.message_id, &message).await;
                return Err(message);
            }
        };
        let mut vector = response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| "embedding backend returned no conversation vector".to_string())?;
        validate_vector(&vector, None)?;
        normalize(&mut vector)?;

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("failed to save conversation embedding: {error}"))?;
        let still_current: i64 = sqlx::query_scalar(
            r#"
            SELECT EXISTS(
                SELECT 1
                FROM conversation_search_documents documents
                JOIN app_settings settings ON settings.id = 1
                WHERE documents.message_id = ?
                  AND documents.revision_id = ?
                  AND settings.notes_semantic_search_enabled = 1
                  AND settings.notes_embedding_backend_id = ?
                  AND settings.notes_embedding_model = ?
            )
            "#,
        )
        .bind(&job.message_id)
        .bind(&job.revision_id)
        .bind(&config.backend_id)
        .bind(&config.model)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("failed to verify conversation embedding source: {error}"))?;
        if still_current == 0 {
            tx.rollback().await.ok();
            return Ok(JobResult::Indexed);
        }
        sqlx::query(
            r#"
            INSERT INTO conversation_embeddings (
                message_id, user_id, revision_id, embedding_backend_id,
                embedding_model, embedding_dimensions, vector, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                user_id = excluded.user_id,
                revision_id = excluded.revision_id,
                embedding_backend_id = excluded.embedding_backend_id,
                embedding_model = excluded.embedding_model,
                embedding_dimensions = excluded.embedding_dimensions,
                vector = excluded.vector,
                indexed_at = excluded.indexed_at
            "#,
        )
        .bind(&job.message_id)
        .bind(&job.user_id)
        .bind(&job.revision_id)
        .bind(&config.backend_id)
        .bind(&config.model)
        .bind(vector.len() as i64)
        .bind(vector_to_bytes(&vector))
        .bind(now)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("failed to insert conversation embedding: {error}"))?;
        sqlx::query(
            "DELETE FROM conversation_embedding_jobs WHERE message_id = ? AND revision_id = ?",
        )
        .bind(&job.message_id)
        .bind(&job.revision_id)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("failed to finish conversation embedding job: {error}"))?;
        sqlx::query(
            r#"
            INSERT INTO conversation_embedding_revisions (user_id, revision)
            VALUES (?, 1)
            ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1
            "#,
        )
        .bind(&job.user_id)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("failed to publish conversation index revision: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("failed to commit conversation embedding: {error}"))?;
        Ok(JobResult::Indexed)
    }

    async fn repair_index(&self) -> Result<(), sqlx::Error> {
        self.sync_documents().await?;
        let Some(config) = self.active_config().await.map_err(sqlx::Error::Protocol)? else {
            return Ok(());
        };
        sqlx::query(
            r#"
            INSERT INTO conversation_embedding_jobs (
                message_id, user_id, revision_id, requested_at,
                attempts, last_attempt_at, last_error
            )
            SELECT documents.message_id, documents.user_id, documents.revision_id,
                   documents.source_updated_at, 0, NULL, NULL
            FROM conversation_search_documents documents
            WHERE NOT EXISTS (
                SELECT 1 FROM conversation_embeddings embeddings
                WHERE embeddings.message_id = documents.message_id
                  AND embeddings.revision_id = documents.revision_id
                  AND embeddings.embedding_backend_id = ?
                  AND embeddings.embedding_model = ?
            )
            ON CONFLICT(message_id) DO UPDATE SET
                user_id = excluded.user_id,
                revision_id = excluded.revision_id,
                requested_at = excluded.requested_at
            "#,
        )
        .bind(config.backend_id)
        .bind(config.model)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn sync_documents(&self) -> Result<(), sqlx::Error> {
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            r#"
            DELETE FROM conversation_search_documents
            WHERE NOT EXISTS (
                SELECT 1
                FROM chat_messages messages
                JOIN chats ON chats.id = messages.chat_id
                JOIN chat_message_revisions revisions
                  ON revisions.id = messages.active_revision_id
                WHERE messages.id = conversation_search_documents.message_id
                  AND messages.role IN ('user', 'assistant')
                  AND messages.status = 'complete'
                  AND messages.is_deleted = 0
                  AND trim(revisions.content_text) != ''
            )
            "#,
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            r#"
            INSERT INTO conversation_search_documents (
                message_id, chat_id, user_id, revision_id, chat_title,
                role, content, source_updated_at, indexed_at
            )
            SELECT messages.id, messages.chat_id, chats.user_id, revisions.id, chats.title,
                   messages.role, revisions.content_text, messages.updated_at, unixepoch()
            FROM chat_messages messages
            JOIN chats ON chats.id = messages.chat_id
            JOIN chat_message_revisions revisions ON revisions.id = messages.active_revision_id
            WHERE messages.role IN ('user', 'assistant')
              AND messages.status = 'complete'
              AND messages.is_deleted = 0
              AND trim(revisions.content_text) != ''
            ON CONFLICT(message_id) DO UPDATE SET
                chat_id = excluded.chat_id,
                user_id = excluded.user_id,
                revision_id = excluded.revision_id,
                chat_title = excluded.chat_title,
                role = excluded.role,
                content = excluded.content,
                source_updated_at = excluded.source_updated_at,
                indexed_at = excluded.indexed_at
            WHERE conversation_search_documents.chat_id != excluded.chat_id
               OR conversation_search_documents.user_id != excluded.user_id
               OR conversation_search_documents.revision_id != excluded.revision_id
               OR conversation_search_documents.chat_title != excluded.chat_title
               OR conversation_search_documents.role != excluded.role
               OR conversation_search_documents.content != excluded.content
               OR conversation_search_documents.source_updated_at != excluded.source_updated_at
            "#,
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
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

    async fn record_failure(&self, message_id: &str, error: &str) {
        let now = unix_timestamp();
        if let Err(db_error) = sqlx::query(
            r#"
            UPDATE conversation_embedding_jobs
            SET attempts = attempts + 1, last_attempt_at = ?, last_error = ?
            WHERE message_id = ?
            "#,
        )
        .bind(now)
        .bind(truncate(error, 500))
        .bind(message_id)
        .execute(&self.pool)
        .await
        {
            tracing::warn!(?db_error, "failed to record conversation embedding error");
        }
    }
}

pub async fn index_status(pool: &SqlitePool) -> Result<(i64, i64, Option<String>), sqlx::Error> {
    let indexed = sqlx::query_scalar("SELECT COUNT(*) FROM conversation_embeddings")
        .fetch_one(pool)
        .await?;
    let pending = sqlx::query_scalar("SELECT COUNT(*) FROM conversation_embedding_jobs")
        .fetch_one(pool)
        .await?;
    let last_error = sqlx::query_scalar(
        r#"
        SELECT last_error FROM conversation_embedding_jobs
        WHERE last_error IS NOT NULL
        ORDER BY last_attempt_at DESC LIMIT 1
        "#,
    )
    .fetch_optional(pool)
    .await?
    .flatten();
    Ok((indexed, pending, last_error))
}

fn row_to_search_result(
    row: sqlx::sqlite::SqliteRow,
) -> Result<ConversationSearchResult, sqlx::Error> {
    Ok(ConversationSearchResult {
        chat_id: row.try_get("chat_id")?,
        message_id: row.try_get("message_id")?,
        chat_title: row.try_get("chat_title")?,
        role: row.try_get("role")?,
        excerpt: truncate(
            &row.try_get::<String, _>("excerpt")?,
            MAX_RESULT_EXCERPT_CHARS,
        ),
        created_at: row.try_get("created_at")?,
    })
}

fn embedding_input(chat_title: &str, role: &str, content: &str) -> String {
    truncate(
        &format!("Chat: {chat_title}\n{role}: {content}"),
        MAX_EMBEDDING_CHARS,
    )
}

fn build_fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .take(16)
        .filter_map(|token| {
            let normalized = token
                .chars()
                .filter(|character| character.is_alphanumeric())
                .collect::<String>();
            (!normalized.is_empty()).then(|| format!("\"{}\"*", normalized.replace('"', "")))
        })
        .collect::<Vec<_>>()
        .join(" AND ")
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
        return Err("stored conversation embedding has an invalid byte length".to_string());
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
        result.push_str("...");
    }
    result
}

fn db_value_error(error: sqlx::Error) -> String {
    format!("invalid conversation index database value: {error}")
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
    use crate::startup;

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

    async fn seed_owner_and_backend(pool: &SqlitePool) {
        sqlx::query(
            r#"
            INSERT INTO users (
                id, username, password_hash, role, is_disabled, created_at, updated_at
            ) VALUES ('owner', 'owner', 'hash', 'admin', 0, 1, 1),
                     ('other', 'other', 'hash', 'user', 0, 1, 1)
            "#,
        )
        .execute(pool)
        .await
        .expect("insert users");
        sqlx::query(
            r#"
            INSERT INTO ollama_backends (
                id, name, base_url, is_enabled, created_at, updated_at
            ) VALUES ('backend', 'Test', 'http://127.0.0.1:11434', 1, 1, 1)
            "#,
        )
        .execute(pool)
        .await
        .expect("insert backend");
        for user_id in ["owner", "other"] {
            sqlx::query(
                r#"
                INSERT INTO user_memory_settings (
                    user_id, allow_model_read, allow_model_create, allow_model_edit,
                    allow_model_forget, allow_model_chat_history, updated_at
                ) VALUES (?, 1, 1, 1, 1, 1, 1)
                "#,
            )
            .bind(user_id)
            .execute(pool)
            .await
            .expect("enable history access");
        }
    }

    async fn insert_message(
        pool: &SqlitePool,
        id: &str,
        revision_id: &str,
        parent_id: Option<&str>,
        role: &str,
        content: &str,
        created_at: i64,
    ) {
        sqlx::query(
            r#"
            INSERT INTO chat_messages (
                id, chat_id, parent_message_id, active_revision_id, role,
                status, is_deleted, created_at, updated_at
            ) VALUES (?, 'chat', ?, ?, ?, 'complete', 0, ?, ?)
            "#,
        )
        .bind(id)
        .bind(parent_id)
        .bind(revision_id)
        .bind(role)
        .bind(created_at)
        .bind(created_at)
        .execute(pool)
        .await
        .expect("insert message");
        sqlx::query(
            r#"
            INSERT INTO chat_message_revisions (
                id, message_id, content_text, thinking_text, source, created_at
            ) VALUES (?, ?, ?, '', 'original', ?)
            "#,
        )
        .bind(revision_id)
        .bind(id)
        .bind(content)
        .bind(created_at)
        .execute(pool)
        .await
        .expect("insert revision");
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
                if text.contains("beach") || text.contains("ocean") {
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
    fn fts_query_is_bounded_and_escaped() {
        assert_eq!(
            build_fts_query("old beach plans"),
            "\"old\"* AND \"beach\"* AND \"plans\"*"
        );
        assert_eq!(build_fts_query("***"), "");
        assert_eq!(
            build_fts_query(&"word ".repeat(40)).matches("word").count(),
            16
        );
    }

    #[test]
    fn embedding_input_is_bounded() {
        let input = embedding_input("A chat", "assistant", &"x".repeat(20_000));
        assert!(input.chars().count() <= MAX_EMBEDDING_CHARS + 3);
        assert!(input.starts_with("Chat: A chat\nassistant: "));
    }

    #[tokio::test]
    async fn knowledge_rebuilds_use_valid_sqlite_upserts() {
        let pool = test_pool().await;
        ConversationRetrieval::new(pool.clone())
            .rebuild_all()
            .await
            .expect("rebuild conversation embeddings");
        crate::memories::retrieval::MemoryRetrieval::new(pool)
            .rebuild_all()
            .await
            .expect("rebuild memory embeddings");
    }

    #[tokio::test]
    async fn current_revisions_are_searchable_and_owner_scoped() {
        let pool = test_pool().await;
        seed_owner_and_backend(&pool).await;
        sqlx::query(
            r#"
            INSERT INTO chats (
                id, user_id, default_backend_id, default_model_name, title,
                created_at, updated_at, last_message_at
            ) VALUES ('chat', 'owner', 'backend', 'test', 'Weekend plans', 1, 1, 1)
            "#,
        )
        .execute(&pool)
        .await
        .expect("insert chat");
        insert_message(
            &pool,
            "user-message",
            "user-r1",
            None,
            "user",
            "Meet at the quiet beach tomorrow.",
            2,
        )
        .await;
        insert_message(
            &pool,
            "assistant-message",
            "assistant-r1",
            Some("user-message"),
            "assistant",
            "Bring the striped umbrella.",
            3,
        )
        .await;

        let retrieval = ConversationRetrieval::new(pool.clone());
        let client = reqwest::Client::new();
        let matches = retrieval
            .hybrid_search(&client, "owner", "quiet beach", 5)
            .await
            .expect("search owner history");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].message_id, "user-message");
        assert!(
            retrieval
                .hybrid_search(&client, "other", "quiet beach", 5)
                .await
                .expect("search other history")
                .is_empty()
        );

        let context = retrieval
            .read_context("owner", "chat", "assistant-message", 8)
            .await
            .expect("read matching branch");
        assert_eq!(context.messages.len(), 2);
        assert_eq!(context.messages[0].message_id, "user-message");
        assert_eq!(context.messages[1].message_id, "assistant-message");
        assert!(
            retrieval
                .read_context("other", "chat", "assistant-message", 8)
                .await
                .is_err()
        );

        sqlx::query(
            r#"
            INSERT INTO chat_message_revisions (
                id, message_id, content_text, thinking_text, source, created_at
            ) VALUES ('user-r2', 'user-message', 'Meet at the mountain lodge.', '', 'edit', 4)
            "#,
        )
        .execute(&pool)
        .await
        .expect("insert edit");
        sqlx::query(
            "UPDATE chat_messages SET active_revision_id = 'user-r2', updated_at = 4 WHERE id = 'user-message'",
        )
        .execute(&pool)
        .await
        .expect("activate edit");
        assert!(
            retrieval
                .hybrid_search(&client, "owner", "quiet beach", 5)
                .await
                .expect("search old revision")
                .is_empty()
        );
        assert_eq!(
            retrieval
                .hybrid_search(&client, "owner", "mountain lodge", 5)
                .await
                .expect("search current revision")[0]
                .message_id,
            "user-message"
        );

        sqlx::query(
            "UPDATE chat_messages SET is_deleted = 1, updated_at = 5 WHERE id = 'user-message'",
        )
        .execute(&pool)
        .await
        .expect("delete message");
        assert!(
            retrieval
                .hybrid_search(&client, "owner", "mountain lodge", 5)
                .await
                .expect("search deleted message")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn model_access_is_denied_by_default() {
        let pool = test_pool().await;
        seed_owner_and_backend(&pool).await;
        sqlx::query("DELETE FROM user_memory_settings WHERE user_id = 'owner'")
            .execute(&pool)
            .await
            .expect("restore default settings");
        let retrieval = ConversationRetrieval::new(pool);
        let error = retrieval
            .hybrid_search(&reqwest::Client::new(), "owner", "anything", 5)
            .await
            .expect_err("history should be opt-in");
        assert_eq!(error.code(), "conversation_search_disabled");
    }

    #[tokio::test]
    async fn semantic_search_finds_differently_worded_history() {
        let pool = test_pool().await;
        seed_owner_and_backend(&pool).await;
        let base_url = fake_embedding_server().await;
        sqlx::query("UPDATE ollama_backends SET base_url = ? WHERE id = 'backend'")
            .bind(base_url)
            .execute(&pool)
            .await
            .expect("point backend at fake server");
        sqlx::query(
            r#"
            UPDATE app_settings
            SET notes_semantic_search_enabled = 1,
                notes_embedding_backend_id = 'backend',
                notes_embedding_model = 'fake-embed'
            WHERE id = 1
            "#,
        )
        .execute(&pool)
        .await
        .expect("enable semantic search");
        sqlx::query(
            r#"
            INSERT INTO chats (
                id, user_id, default_backend_id, default_model_name, title,
                created_at, updated_at, last_message_at
            ) VALUES ('chat', 'owner', 'backend', 'test', 'Weekend plans', 1, 1, 1)
            "#,
        )
        .execute(&pool)
        .await
        .expect("insert chat");
        insert_message(
            &pool,
            "user-message",
            "user-r1",
            None,
            "user",
            "Meet at the quiet beach tomorrow.",
            2,
        )
        .await;

        let retrieval = ConversationRetrieval::new(pool.clone());
        retrieval.repair_index().await.expect("repair queue");
        assert!(matches!(
            retrieval
                .process_next_job(&reqwest::Client::new())
                .await
                .expect("index message"),
            JobResult::Indexed
        ));
        let matches = retrieval
            .hybrid_search(&reqwest::Client::new(), "owner", "ocean meeting", 5)
            .await
            .expect("semantic search");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].message_id, "user-message");
    }
}
