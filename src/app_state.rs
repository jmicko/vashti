use std::{collections::HashMap, sync::Arc};

use sqlx::SqlitePool;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::config::Config;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub db: SqlitePool,
    pub http_client: reqwest::Client,
    pub generation_cancellations: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl AppState {
    pub fn new(config: Config, db: SqlitePool, http_client: reqwest::Client) -> Self {
        Self {
            config: Arc::new(config),
            db,
            http_client,
            generation_cancellations: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}
