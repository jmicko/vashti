pub mod bootstrap;
pub mod migrations;
pub mod network_recovery;

use crate::config::Config;

pub async fn prepare_data_dir(config: &Config) -> Result<(), std::io::Error> {
    tokio::fs::create_dir_all(&config.data_dir).await?;
    tokio::fs::create_dir_all(config.uploads_dir()).await?;
    tokio::fs::create_dir_all(config.tmp_dir()).await?;
    Ok(())
}
