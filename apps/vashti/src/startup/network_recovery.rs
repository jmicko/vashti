use std::path::{Path, PathBuf};

use sqlx::SqlitePool;

use crate::{auth::service::unix_timestamp, config::Config, settings};

const RECOVERY_FILE: &str = "recover_network.txt";
const SUCCESS_FILE: &str = "recover_network_success.txt";

pub async fn recover_network_if_requested(
    pool: &SqlitePool,
    config: &Config,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let recovery_path = config.app_root.join(RECOVERY_FILE);
    if tokio::fs::metadata(&recovery_path).await.is_err() {
        return Ok(());
    }

    let success_path = next_success_path(&config.app_root).await;
    let notice = format!(
        "Network settings were reset because {RECOVERY_FILE} was found. The file was renamed to {}.",
        success_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(SUCCESS_FILE)
    );

    settings::service::reset_network_settings_for_recovery(pool, &notice).await?;
    tokio::fs::rename(&recovery_path, &success_path).await?;
    tracing::warn!(
        ?recovery_path,
        ?success_path,
        "network settings recovery applied"
    );

    Ok(())
}

async fn next_success_path(app_root: &Path) -> PathBuf {
    let base = app_root.join(SUCCESS_FILE);
    if tokio::fs::metadata(&base).await.is_err() {
        return base;
    }

    app_root.join(format!("recover_network_success_{}.txt", unix_timestamp()))
}
