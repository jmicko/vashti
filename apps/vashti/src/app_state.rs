use std::{collections::HashMap, sync::Arc};

use sqlx::SqlitePool;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::{
    client_tools::ClientToolBroker, config::Config, model_cache::ModelCache,
    model_pulls::ModelPullManager, notes::retrieval::NoteRetrieval, rate_limit::RateLimiter,
    updates::UpdateManager,
};

#[derive(Clone, Debug)]
pub struct GenerationProgress {
    pub user_id: String,
    pub chat_id: String,
    pub content_text: String,
    pub thinking_text: String,
}

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub db: SqlitePool,
    pub http_client: reqwest::Client,
    pub server_instance_id: Arc<str>,
    pub rate_limiter: Arc<RateLimiter>,
    pub model_cache: Arc<ModelCache>,
    pub model_pulls: Arc<ModelPullManager>,
    pub note_retrieval: Arc<NoteRetrieval>,
    pub updates: Arc<UpdateManager>,
    pub client_tools: ClientToolBroker,
    pub generation_cancellations: Arc<Mutex<HashMap<String, CancellationToken>>>,
    pub generation_progress: Arc<Mutex<HashMap<String, GenerationProgress>>>,
}

impl AppState {
    pub fn new(
        config: Config,
        db: SqlitePool,
        http_client: reqwest::Client,
        server_instance_id: String,
    ) -> Self {
        Self {
            config: Arc::new(config),
            note_retrieval: Arc::new(NoteRetrieval::new(db.clone())),
            db,
            http_client,
            server_instance_id: server_instance_id.into(),
            rate_limiter: Arc::new(RateLimiter::new()),
            model_cache: Arc::new(ModelCache::new()),
            model_pulls: Arc::new(ModelPullManager::new()),
            updates: Arc::new(UpdateManager::new()),
            client_tools: ClientToolBroker::default(),
            generation_cancellations: Arc::new(Mutex::new(HashMap::new())),
            generation_progress: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}
