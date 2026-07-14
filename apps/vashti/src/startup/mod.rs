pub mod bootstrap;
pub mod migrations;
pub mod network_recovery;

use crate::config::Config;

pub async fn prepare_data_dir(config: &Config) -> Result<(), std::io::Error> {
    tokio::fs::create_dir_all(&config.data_dir).await?;
    tokio::fs::create_dir_all(config.uploads_dir()).await?;
    tokio::fs::create_dir_all(config.persona_avatars_dir()).await?;
    tokio::fs::create_dir_all(config.tmp_dir()).await?;

    secure_directory(&config.data_dir).await?;
    secure_directory(&config.uploads_dir()).await?;
    secure_directory(&config.persona_avatars_dir()).await?;
    secure_directory(&config.tmp_dir()).await?;
    Ok(())
}

pub async fn secure_data_files(config: &Config) -> Result<(), std::io::Error> {
    secure_file_if_present(&config.database_path).await?;
    secure_file_if_present(&config.database_path.with_extension("db-wal")).await?;
    secure_file_if_present(&config.database_path.with_extension("db-shm")).await?;
    Ok(())
}

#[cfg(unix)]
async fn secure_directory(path: &std::path::Path) -> Result<(), std::io::Error> {
    set_mode(path, 0o700).await
}

#[cfg(not(unix))]
async fn secure_directory(_path: &std::path::Path) -> Result<(), std::io::Error> {
    Ok(())
}

#[cfg(unix)]
async fn secure_file_if_present(path: &std::path::Path) -> Result<(), std::io::Error> {
    if tokio::fs::try_exists(path).await? {
        set_mode(path, 0o600).await?;
    }
    Ok(())
}

#[cfg(not(unix))]
async fn secure_file_if_present(_path: &std::path::Path) -> Result<(), std::io::Error> {
    Ok(())
}

#[cfg(unix)]
async fn set_mode(path: &std::path::Path, mode: u32) -> Result<(), std::io::Error> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = tokio::fs::metadata(path).await?;
    let mut permissions = metadata.permissions();
    if permissions.mode() & 0o777 != mode {
        permissions.set_mode(mode);
        tokio::fs::set_permissions(path, permissions).await?;
    }
    Ok(())
}
