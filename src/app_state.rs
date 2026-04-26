use std::sync::Arc;

use sqlx::SqlitePool;

use crate::config::Config;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub db: SqlitePool,
    pub http_client: reqwest::Client,
}

impl AppState {
    pub fn new(config: Config, db: SqlitePool, http_client: reqwest::Client) -> Self {
        Self {
            config: Arc::new(config),
            db,
            http_client,
        }
    }
}
