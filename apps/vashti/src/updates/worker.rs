use std::{
    env,
    fs::{File, OpenOptions},
    io::{self, Read},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Component, Path, PathBuf},
    time::Duration,
};

use flate2::read::GzDecoder;
use futures_util::StreamExt;
use reqwest::StatusCode;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use uuid::Uuid;
use vashti_update_manifest::{SCHEMA_VERSION, SignedArtifact, verify_release_signature};

use super::service::{
    UpdateOperationStatus, UpdateRequest, parse_version, update_target, version_label,
    write_json_atomic,
};

const MAX_UPDATE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 256 * 1024;
const MAX_REQUEST_BYTES: u64 = 4096;
const MAX_ARCHIVE_ENTRIES: usize = 128;
const BACKUPS_TO_KEEP: usize = 3;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(45);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Deserialize)]
struct RemoteUpdateManifest {
    schema_version: u32,
    channel: String,
    release_status: String,
    artifact: SignedArtifact,
    signature: String,
    download_url: String,
}

#[derive(Debug)]
struct WorkerConfig {
    data_dir: PathBuf,
    work_dir: PathBuf,
    binary_path: PathBuf,
    database_path: PathBuf,
    service_name: String,
    health_url: String,
    update_base_url: String,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("managed updater must run as root")]
    RootRequired,
    #[error("managed updater configuration is invalid: {0}")]
    InvalidConfig(String),
    #[error("update request is invalid: {0}")]
    InvalidRequest(String),
    #[error("update manifest is invalid: {0}")]
    InvalidManifest(String),
    #[error("update request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("filesystem operation failed: {0}")]
    Io(#[from] io::Error),
    #[error("JSON operation failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("service operation failed: {0}")]
    Service(String),
    #[error("the update failed health checks and Vashti was rolled back: {0}")]
    RolledBack(String),
}

impl WorkerConfig {
    fn from_env() -> Result<Self, WorkerError> {
        #[cfg(unix)]
        if unsafe { libc::geteuid() } != 0 {
            return Err(WorkerError::RootRequired);
        }

        let data_dir = absolute_env_path("VASHTI_UPDATE_DATA_DIR", "/var/lib/vashti")?;
        let work_dir = absolute_env_path("VASHTI_UPDATE_WORK_DIR", "/var/lib/vashti-update")?;
        ensure_secure_work_dir(&work_dir)?;
        let binary_path = absolute_env_path("VASHTI_UPDATE_BINARY_PATH", "/usr/local/bin/vashti")?;
        let database_path = env::var_os("VASHTI_UPDATE_DATABASE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("app.db"));
        if !database_path.is_absolute() {
            return Err(WorkerError::InvalidConfig(
                "VASHTI_UPDATE_DATABASE_PATH must be absolute".to_string(),
            ));
        }
        let service_name =
            env::var("VASHTI_UPDATE_SERVICE_NAME").unwrap_or_else(|_| "vashti.service".to_string());
        if service_name.is_empty()
            || !service_name.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'@')
            })
        {
            return Err(WorkerError::InvalidConfig(
                "VASHTI_UPDATE_SERVICE_NAME contains unsupported characters".to_string(),
            ));
        }
        let health_url = env::var("VASHTI_UPDATE_HEALTH_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:7771/api/version".to_string());
        validate_url(&health_url, true)?;
        let update_base_url = env::var("VASHTI_UPDATE_BASE_URL")
            .unwrap_or_else(|_| "https://vashti.chat".to_string());
        validate_url(&update_base_url, false)?;

        Ok(Self {
            data_dir,
            work_dir,
            binary_path,
            database_path,
            service_name,
            health_url,
            update_base_url,
        })
    }

    fn update_dir(&self) -> PathBuf {
        self.data_dir.join("update")
    }

    fn request_path(&self) -> PathBuf {
        self.update_dir().join("request.json")
    }

    fn processing_request_path(&self) -> PathBuf {
        self.update_dir().join("request.in-progress.json")
    }

    fn last_request_path(&self) -> PathBuf {
        self.update_dir().join("last-request.json")
    }

    fn status_path(&self) -> PathBuf {
        self.update_dir().join("status.json")
    }

    fn backups_dir(&self) -> PathBuf {
        self.work_dir.join("backups")
    }
}

pub async fn apply_requested_update() -> Result<(), WorkerError> {
    let config = WorkerConfig::from_env()?;
    tokio::fs::create_dir_all(config.update_dir()).await?;
    cleanup_abandoned_staging(&config).await;
    let request = match claim_request(&config).await {
        Ok(request) => request,
        Err(error) => {
            let message = error.to_string();
            if let Err(status_error) = write_status(&config, "failed", None, &message).await {
                tracing::error!(?status_error, "failed to record an invalid update request");
            }
            finish_request(&config).await;
            return Err(error);
        }
    };

    let result = async {
        write_status(
            &config,
            "installing",
            Some(&request.version),
            "Downloading and verifying the update before restart.",
        )
        .await?;
        perform_update(&config, &request).await
    }
    .await;
    let (state, message) = match &result {
        Ok(()) => (
            "succeeded",
            format!("Vashti {} was installed successfully.", request.version),
        ),
        Err(WorkerError::RolledBack(message)) => ("rolled_back", message.clone()),
        Err(error) => ("failed", error.to_string()),
    };
    if let Err(error) = write_status(&config, state, Some(&request.version), &message).await {
        tracing::error!(?error, "failed to write final managed update status");
    }
    finish_request(&config).await;
    result
}

async fn perform_update(config: &WorkerConfig, request: &UpdateRequest) -> Result<(), WorkerError> {
    validate_request(request)?;
    let download_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(10 * 60))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let health_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(3))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let prepared = prepare_update(config, request, &download_client).await?;
    let result = install_prepared_update(config, request, &health_client, &prepared).await;
    if let Err(error) = tokio::fs::remove_dir_all(&prepared.staging_dir).await
        && error.kind() != io::ErrorKind::NotFound
    {
        tracing::warn!(?error, "failed to remove managed update staging directory");
    }
    result
}

async fn install_prepared_update(
    config: &WorkerConfig,
    request: &UpdateRequest,
    client: &reqwest::Client,
    prepared: &PreparedUpdate,
) -> Result<(), WorkerError> {
    let current_version = binary_version(&config.binary_path).await?;
    if parse_version(&request.version).map_err(map_update_error)?
        <= parse_version(&current_version).map_err(map_update_error)?
    {
        return Err(WorkerError::InvalidRequest(
            "managed updates refuse reinstalls and downgrades".to_string(),
        ));
    }

    systemctl(config, "stop").await?;
    let backup_dir = match create_backup(config, &current_version).await {
        Ok(path) => path,
        Err(error) => {
            let _ = systemctl(config, "start").await;
            return Err(error);
        }
    };

    if let Err(error) = install_binary(config, &prepared.binary_path).await {
        let _ = systemctl(config, "start").await;
        return Err(error);
    }

    let start_result = systemctl(config, "start").await;
    let health_result = match start_result {
        Ok(()) => wait_for_version(client, &config.health_url, &request.version).await,
        Err(error) => Err(error),
    };
    if let Err(error) = health_result {
        rollback(config, &backup_dir, &current_version, client).await?;
        prune_backups(config, BACKUPS_TO_KEEP).await;
        return Err(WorkerError::RolledBack(format!(
            "The update failed its health check and Vashti was restored to {}: {error}",
            version_label(&current_version)
        )));
    }

    prune_backups(config, BACKUPS_TO_KEEP).await;
    Ok(())
}

struct PreparedUpdate {
    staging_dir: PathBuf,
    binary_path: PathBuf,
}

async fn prepare_update(
    config: &WorkerConfig,
    request: &UpdateRequest,
    client: &reqwest::Client,
) -> Result<PreparedUpdate, WorkerError> {
    let target = update_target().ok_or_else(|| {
        WorkerError::InvalidConfig("managed updates do not support this target".to_string())
    })?;
    let base_url = reqwest::Url::parse(&config.update_base_url)
        .map_err(|_| WorkerError::InvalidConfig("update base URL is invalid".to_string()))?;
    let manifest_url = base_url
        .join(&format!("/api/updates/{}/{target}", request.channel))
        .map_err(|_| WorkerError::InvalidConfig("update base URL is invalid".to_string()))?;
    let response = client.get(manifest_url).send().await?;
    if response.status() == StatusCode::NOT_FOUND {
        return Err(WorkerError::InvalidManifest(
            "the requested release is no longer available".to_string(),
        ));
    }
    let manifest = read_remote_manifest(response.error_for_status()?).await?;
    validate_manifest(&manifest, request, target)?;

    let artifact_url = base_url
        .join(&format!(
            "/releases/{}/{}",
            manifest.artifact.version, manifest.artifact.filename
        ))
        .map_err(|_| WorkerError::InvalidConfig("update base URL is invalid".to_string()))?;
    let response = client.get(artifact_url).send().await?.error_for_status()?;
    if response.content_length().is_some_and(|length| {
        length > MAX_UPDATE_BYTES || length != manifest.artifact.size_bytes as u64
    }) {
        return Err(WorkerError::InvalidManifest(
            "update artifact size does not match the signed manifest".to_string(),
        ));
    }

    let staging_dir = config.work_dir.join(format!("staging-{}", Uuid::new_v4()));
    tokio::fs::create_dir_all(&staging_dir).await?;
    let result = async {
        let archive_path = staging_dir.join(&manifest.artifact.filename);
        download_artifact(response, &archive_path, &manifest.artifact).await?;
        let binary_path = staging_dir.join("vashti");
        let archive_path_for_extract = archive_path.clone();
        let binary_path_for_extract = binary_path.clone();
        tokio::task::spawn_blocking(move || {
            extract_binary(&archive_path_for_extract, &binary_path_for_extract)
        })
        .await
        .map_err(|error| WorkerError::InvalidManifest(error.to_string()))??;
        tokio::fs::set_permissions(&binary_path, std::fs::Permissions::from_mode(0o755)).await?;
        let staged_version = binary_version(&binary_path).await?;
        if version_label(&staged_version) != request.version {
            return Err(WorkerError::InvalidManifest(
                "staged binary version does not match the signed release".to_string(),
            ));
        }
        Ok(binary_path)
    }
    .await;

    match result {
        Ok(binary_path) => Ok(PreparedUpdate {
            staging_dir,
            binary_path,
        }),
        Err(error) => {
            if let Err(cleanup_error) = tokio::fs::remove_dir_all(&staging_dir).await {
                tracing::warn!(
                    ?cleanup_error,
                    "failed to clean rejected update staging data"
                );
            }
            Err(error)
        }
    }
}

async fn read_remote_manifest(
    response: reqwest::Response,
) -> Result<RemoteUpdateManifest, WorkerError> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_MANIFEST_BYTES as u64)
    {
        return Err(WorkerError::InvalidManifest(
            "update manifest exceeds the size limit".to_string(),
        ));
    }

    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if bytes.len().saturating_add(chunk.len()) > MAX_MANIFEST_BYTES {
            return Err(WorkerError::InvalidManifest(
                "update manifest exceeds the size limit".to_string(),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }

    serde_json::from_slice(&bytes).map_err(WorkerError::from)
}

async fn download_artifact(
    response: reqwest::Response,
    archive_path: &Path,
    artifact: &SignedArtifact,
) -> Result<(), WorkerError> {
    let mut output = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(archive_path)
        .await?;
    let mut stream = response.bytes_stream();
    let mut hasher = Sha256::new();
    let mut size_bytes = 0_u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        size_bytes = size_bytes.checked_add(chunk.len() as u64).ok_or_else(|| {
            WorkerError::InvalidManifest("update artifact size overflowed".to_string())
        })?;
        if size_bytes > MAX_UPDATE_BYTES || size_bytes > artifact.size_bytes as u64 {
            return Err(WorkerError::InvalidManifest(
                "update artifact exceeds its signed size".to_string(),
            ));
        }
        hasher.update(&chunk);
        output.write_all(&chunk).await?;
    }

    if size_bytes != artifact.size_bytes as u64 {
        return Err(WorkerError::InvalidManifest(
            "update artifact size does not match the signed manifest".to_string(),
        ));
    }
    let actual_sha = format!("{:x}", hasher.finalize());
    if actual_sha != artifact.sha256 {
        return Err(WorkerError::InvalidManifest(
            "update artifact hash does not match the signed manifest".to_string(),
        ));
    }
    output.flush().await?;
    output.sync_all().await?;
    Ok(())
}

fn validate_request(request: &UpdateRequest) -> Result<(), WorkerError> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(WorkerError::InvalidRequest(
            "unsupported request schema".to_string(),
        ));
    }
    if !matches!(request.channel.as_str(), "stable" | "prerelease") {
        return Err(WorkerError::InvalidRequest(
            "unsupported update channel".to_string(),
        ));
    }
    parse_version(&request.version).map_err(map_update_error)?;
    Ok(())
}

fn validate_manifest(
    manifest: &RemoteUpdateManifest,
    request: &UpdateRequest,
    target: &str,
) -> Result<(), WorkerError> {
    if manifest.schema_version != SCHEMA_VERSION
        || manifest.channel != request.channel
        || manifest.artifact.version != request.version
        || manifest.artifact.target != target
    {
        return Err(WorkerError::InvalidManifest(
            "manifest does not match the constrained update request".to_string(),
        ));
    }
    if manifest.artifact.size_bytes as u64 > MAX_UPDATE_BYTES {
        return Err(WorkerError::InvalidManifest(
            "update artifact exceeds the size limit".to_string(),
        ));
    }
    let valid_release_status = match request.channel.as_str() {
        "stable" => manifest.release_status == "released",
        "prerelease" => matches!(manifest.release_status.as_str(), "released" | "prerelease"),
        _ => false,
    };
    if !valid_release_status {
        return Err(WorkerError::InvalidManifest(
            "release status does not match the requested channel".to_string(),
        ));
    }
    let expected_download_url = format!(
        "/releases/{}/{}",
        manifest.artifact.version, manifest.artifact.filename
    );
    if manifest.download_url != expected_download_url {
        return Err(WorkerError::InvalidManifest(
            "manifest artifact path is inconsistent".to_string(),
        ));
    }
    verify_release_signature(&manifest.artifact, &manifest.signature)
        .map_err(|error| WorkerError::InvalidManifest(error.to_string()))
}

fn extract_binary(archive_path: &Path, output_path: &Path) -> Result<(), WorkerError> {
    let archive_file = File::open(archive_path)?;
    let decoder = GzDecoder::new(archive_file);
    let mut archive = tar::Archive::new(decoder);
    let mut found = false;
    let mut entry_count = 0_usize;
    let mut unpacked_bytes = 0_u64;
    for entry in archive.entries()? {
        let mut entry = entry?;
        entry_count += 1;
        let entry_size = entry.header().size()?;
        unpacked_bytes = unpacked_bytes.checked_add(entry_size).ok_or_else(|| {
            WorkerError::InvalidManifest("release archive size overflowed".to_string())
        })?;
        if entry_count > MAX_ARCHIVE_ENTRIES || unpacked_bytes > MAX_UPDATE_BYTES {
            return Err(WorkerError::InvalidManifest(
                "release archive exceeds extraction limits".to_string(),
            ));
        }
        let path = entry.path()?;
        let components = path.components().collect::<Vec<_>>();
        if components.len() != 2
            || !matches!(components[0], Component::Normal(_))
            || components[1].as_os_str() != "vashti"
        {
            continue;
        }
        if !entry.header().entry_type().is_file() || found {
            return Err(WorkerError::InvalidManifest(
                "release archive contains an invalid Vashti binary entry".to_string(),
            ));
        }
        let mut output = File::create(output_path)?;
        io::copy(&mut entry, &mut output)?;
        output.sync_all()?;
        found = true;
    }
    if !found {
        return Err(WorkerError::InvalidManifest(
            "release archive does not contain a Vashti binary".to_string(),
        ));
    }
    Ok(())
}

async fn cleanup_abandoned_staging(config: &WorkerConfig) {
    let Ok(mut entries) = tokio::fs::read_dir(&config.work_dir).await else {
        return;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("staging-")
            && entry.file_type().await.is_ok_and(|kind| kind.is_dir())
            && let Err(error) = tokio::fs::remove_dir_all(entry.path()).await
        {
            tracing::warn!(?error, path = ?entry.path(), "failed to remove stale update staging data");
        }
    }
}

async fn prune_backups(config: &WorkerConfig, keep: usize) {
    let backups_dir = config.backups_dir();
    let Ok(mut entries) = tokio::fs::read_dir(&backups_dir).await else {
        return;
    };
    let mut backups = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        if !entry.file_type().await.is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        let modified = entry
            .metadata()
            .await
            .and_then(|metadata| metadata.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        backups.push((modified, entry.path()));
    }
    backups.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    for (_, path) in backups.into_iter().skip(keep) {
        if let Err(error) = tokio::fs::remove_dir_all(&path).await {
            tracing::warn!(
                ?error,
                ?path,
                "failed to prune an old managed update backup"
            );
        }
    }
}

async fn create_backup(
    config: &WorkerConfig,
    current_version: &str,
) -> Result<PathBuf, WorkerError> {
    let backup_dir = config.backups_dir().join(format!(
        "{}-{}-{}",
        version_label(current_version),
        time::OffsetDateTime::now_utc().unix_timestamp(),
        Uuid::new_v4()
    ));
    tokio::fs::create_dir_all(&backup_dir).await?;
    atomic_copy_preserving_metadata(&config.binary_path, &backup_dir.join("vashti")).await?;
    for source in database_files(&config.database_path) {
        if tokio::fs::try_exists(&source).await? {
            let filename = source.file_name().ok_or_else(|| {
                WorkerError::InvalidConfig("database path has no filename".to_string())
            })?;
            atomic_copy_preserving_metadata(&source, &backup_dir.join(filename)).await?;
        }
    }
    Ok(backup_dir)
}

async fn install_binary(config: &WorkerConfig, staged_binary: &Path) -> Result<(), WorkerError> {
    atomic_copy(staged_binary, &config.binary_path, 0o755).await
}

async fn rollback(
    config: &WorkerConfig,
    backup_dir: &Path,
    previous_version: &str,
    client: &reqwest::Client,
) -> Result<(), WorkerError> {
    let _ = systemctl(config, "stop").await;
    atomic_copy_preserving_metadata(&backup_dir.join("vashti"), &config.binary_path).await?;

    for destination in database_files(&config.database_path) {
        match tokio::fs::remove_file(&destination).await {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    for destination in database_files(&config.database_path) {
        let filename = destination.file_name().ok_or_else(|| {
            WorkerError::InvalidConfig("database path has no filename".to_string())
        })?;
        let source = backup_dir.join(filename);
        if tokio::fs::try_exists(&source).await? {
            atomic_copy_preserving_metadata(&source, &destination).await?;
        }
    }

    systemctl(config, "start").await?;
    wait_for_version(client, &config.health_url, previous_version).await
}

async fn atomic_copy(source: &Path, destination: &Path, mode: u32) -> Result<(), WorkerError> {
    atomic_copy_inner(source, destination, Some(mode)).await
}

async fn atomic_copy_preserving_metadata(
    source: &Path,
    destination: &Path,
) -> Result<(), WorkerError> {
    atomic_copy_inner(source, destination, None).await
}

async fn atomic_copy_inner(
    source: &Path,
    destination: &Path,
    fixed_mode: Option<u32>,
) -> Result<(), WorkerError> {
    let parent = destination.parent().ok_or_else(|| {
        WorkerError::InvalidConfig("update destination has no parent".to_string())
    })?;
    let temporary = parent.join(format!(".vashti-update-{}", Uuid::new_v4()));
    let source = source.to_path_buf();
    let destination = destination.to_path_buf();
    let parent = parent.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let copy_result = (|| {
            let mut input = File::open(&source)?;
            let source_metadata = input.metadata()?;
            let mut output = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(fixed_mode.unwrap_or(0o600))
                .open(&temporary)?;
            io::copy(&mut input, &mut output)?;
            if fixed_mode.is_none() {
                let output_metadata = output.metadata()?;
                if output_metadata.uid() != source_metadata.uid()
                    || output_metadata.gid() != source_metadata.gid()
                {
                    let result = unsafe {
                        libc::fchown(
                            output.as_raw_fd(),
                            source_metadata.uid(),
                            source_metadata.gid(),
                        )
                    };
                    if result != 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
            }
            let mode = fixed_mode.unwrap_or(source_metadata.mode() & 0o7777);
            output.set_permissions(std::fs::Permissions::from_mode(mode))?;
            output.sync_all()?;
            drop(output);
            std::fs::rename(&temporary, &destination)?;
            Ok::<(), io::Error>(())
        })();
        if copy_result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        copy_result?;
        if let Err(error) = File::open(parent).and_then(|directory| directory.sync_all()) {
            tracing::warn!(
                ?error,
                ?destination,
                "atomic update file was committed but its directory did not sync"
            );
        }
        Ok::<(), io::Error>(())
    })
    .await
    .map_err(|error| WorkerError::Io(io::Error::other(error)))??;
    Ok(())
}

async fn binary_version(path: &Path) -> Result<String, WorkerError> {
    let mut command = Command::new(path);
    command.arg("--version").kill_on_drop(true);
    let output = tokio::time::timeout(COMMAND_TIMEOUT, command.output())
        .await
        .map_err(|_| {
            WorkerError::InvalidManifest("binary version check timed out".to_string())
        })??;
    if !output.status.success() {
        return Err(WorkerError::InvalidManifest(
            "Vashti binary did not report a version".to_string(),
        ));
    }
    let version = String::from_utf8(output.stdout)
        .map_err(|_| WorkerError::InvalidManifest("binary version is not UTF-8".to_string()))?;
    let version = version.trim().to_string();
    parse_version(&version).map_err(map_update_error)?;
    Ok(version)
}

async fn systemctl(config: &WorkerConfig, action: &str) -> Result<(), WorkerError> {
    let mut command = Command::new("systemctl");
    command
        .arg(action)
        .arg(&config.service_name)
        .kill_on_drop(true);
    let output = tokio::time::timeout(COMMAND_TIMEOUT, command.output())
        .await
        .map_err(|_| WorkerError::Service(format!("systemctl {action} timed out")))??;
    if output.status.success() {
        return Ok(());
    }
    Err(WorkerError::Service(format!(
        "systemctl {action} failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    )))
}

async fn wait_for_version(
    client: &reqwest::Client,
    health_url: &str,
    expected_version: &str,
) -> Result<(), WorkerError> {
    let deadline = tokio::time::Instant::now() + HEALTH_TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        if let Ok(response) = client.get(health_url).send().await
            && let Ok(response) = response.error_for_status()
            && let Ok(body) = response.json::<serde_json::Value>().await
            && body.get("version").and_then(serde_json::Value::as_str)
                == Some(expected_version.trim_start_matches('v'))
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(WorkerError::Service(format!(
        "Vashti did not report version {} within {} seconds",
        version_label(expected_version),
        HEALTH_TIMEOUT.as_secs()
    )))
}

async fn claim_request(config: &WorkerConfig) -> Result<UpdateRequest, WorkerError> {
    if !tokio::fs::try_exists(config.processing_request_path()).await? {
        tokio::fs::rename(config.request_path(), config.processing_request_path()).await?;
    }
    let path = config.processing_request_path();
    let bytes = tokio::task::spawn_blocking(move || read_bounded_regular_file(&path))
        .await
        .map_err(|error| WorkerError::Io(io::Error::other(error)))??;
    let request = serde_json::from_slice::<UpdateRequest>(&bytes)?;
    validate_request(&request)?;
    Ok(request)
}

fn read_bounded_regular_file(path: &Path) -> Result<Vec<u8>, WorkerError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_REQUEST_BYTES {
        return Err(WorkerError::InvalidRequest(
            "update request must be a small regular file".to_string(),
        ));
    }

    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_REQUEST_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_REQUEST_BYTES {
        return Err(WorkerError::InvalidRequest(
            "update request exceeds the size limit".to_string(),
        ));
    }
    Ok(bytes)
}

async fn finish_request(config: &WorkerConfig) {
    if let Err(error) =
        tokio::fs::rename(config.processing_request_path(), config.last_request_path()).await
    {
        tracing::warn!(?error, "failed to archive managed update request");
    }
}

async fn write_status(
    config: &WorkerConfig,
    state: &str,
    version: Option<&str>,
    message: &str,
) -> Result<(), WorkerError> {
    write_json_atomic(
        &config.status_path(),
        &UpdateOperationStatus {
            state: state.to_string(),
            version: version.map(str::to_string),
            message: Some(message.to_string()),
            updated_at: Some(time::OffsetDateTime::now_utc().unix_timestamp()),
        },
    )
    .await
    .map_err(|error| WorkerError::InvalidConfig(error.to_string()))
}

fn database_files(database_path: &Path) -> [PathBuf; 3] {
    let value = database_path.to_string_lossy();
    [
        database_path.to_path_buf(),
        PathBuf::from(format!("{value}-wal")),
        PathBuf::from(format!("{value}-shm")),
    ]
}

fn absolute_env_path(name: &str, default: &str) -> Result<PathBuf, WorkerError> {
    let path = env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    if !path.is_absolute() {
        return Err(WorkerError::InvalidConfig(format!(
            "{name} must be absolute"
        )));
    }
    Ok(path)
}

fn ensure_secure_work_dir(path: &Path) -> Result<(), WorkerError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(WorkerError::InvalidConfig(
                "managed update work directory must be a real directory".to_string(),
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            std::fs::create_dir_all(path)?;
        }
        Err(error) => return Err(error.into()),
    }

    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.uid() != 0 {
        return Err(WorkerError::InvalidConfig(
            "managed update work directory must be owned by root".to_string(),
        ));
    }
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    let mode = std::fs::symlink_metadata(path)?.mode();
    if mode & 0o077 != 0 {
        return Err(WorkerError::InvalidConfig(
            "managed update work directory must not be accessible by other users".to_string(),
        ));
    }
    Ok(())
}

fn validate_url(value: &str, health_check: bool) -> Result<(), WorkerError> {
    let url = reqwest::Url::parse(value)
        .map_err(|_| WorkerError::InvalidConfig("configured URL is invalid".to_string()))?;
    if url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(WorkerError::InvalidConfig(
            "configured URL must not contain credentials, a query, or a fragment".to_string(),
        ));
    }
    if health_check {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(WorkerError::InvalidConfig(
                "health check URL must use HTTP or HTTPS".to_string(),
            ));
        }
    } else {
        let local_http = url.scheme() == "http"
            && url
                .host_str()
                .is_some_and(|host| matches!(host, "localhost" | "127.0.0.1" | "::1"));
        if (!local_http && url.scheme() != "https") || url.path() != "/" {
            return Err(WorkerError::InvalidConfig(
                "update base URL must be an HTTPS origin or a loopback HTTP origin without a path"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

fn map_update_error(error: super::service::UpdateError) -> WorkerError {
    WorkerError::InvalidManifest(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, os::unix::fs::symlink};

    use flate2::{Compression, write::GzEncoder};
    use tar::{Builder, EntryType, Header};

    use super::*;

    const FIXTURE_SIGNATURE: &str =
        "hQJFKgrhovM2Sa9XtxaNgftzQosjzge6uUPDZBYf97/sIYabYPVqilGNc+L6K/dhB92A+dhhTPvCztd13NO9CQ==";

    fn fixture_artifact() -> SignedArtifact {
        SignedArtifact {
            version: "v1.2.3".to_string(),
            target: "linux-x86_64".to_string(),
            filename: "vashti-linux-x86_64.tar.gz".to_string(),
            sha256: "a".repeat(64),
            size_bytes: 42,
        }
    }

    fn fixture_manifest(channel: &str, release_status: &str) -> RemoteUpdateManifest {
        RemoteUpdateManifest {
            schema_version: SCHEMA_VERSION,
            channel: channel.to_string(),
            release_status: release_status.to_string(),
            artifact: fixture_artifact(),
            signature: FIXTURE_SIGNATURE.to_string(),
            download_url: "/releases/v1.2.3/vashti-linux-x86_64.tar.gz".to_string(),
        }
    }

    fn fixture_request(channel: &str) -> UpdateRequest {
        UpdateRequest {
            schema_version: SCHEMA_VERSION,
            channel: channel.to_string(),
            version: "v1.2.3".to_string(),
            requested_at: 1,
        }
    }

    fn test_directory(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("vashti-update-{name}-{}", Uuid::new_v4()))
    }

    fn append_regular_file(builder: &mut Builder<GzEncoder<File>>, path: &str, contents: &[u8]) {
        let mut header = Header::new_gnu();
        header.set_entry_type(EntryType::Regular);
        header.set_mode(0o755);
        header.set_size(contents.len() as u64);
        header.set_cksum();
        builder
            .append_data(&mut header, path, Cursor::new(contents))
            .expect("test archive entry should be written");
    }

    fn write_archive(path: &Path, entries: &[(&str, &[u8])]) {
        let encoder = GzEncoder::new(
            File::create(path).expect("test archive should be created"),
            Compression::default(),
        );
        let mut builder = Builder::new(encoder);
        for (entry_path, contents) in entries {
            append_regular_file(&mut builder, entry_path, contents);
        }
        builder.finish().expect("test archive should finish");
    }

    #[test]
    fn service_names_reject_shell_metacharacters() {
        assert!(
            "vashti.service"
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric()
                    || matches!(byte, b'.' | b'_' | b'-' | b'@'))
        );
        assert!(
            !"vashti.service;reboot"
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric()
                    || matches!(byte, b'.' | b'_' | b'-' | b'@'))
        );
    }

    #[test]
    fn remote_update_url_requires_https() {
        assert!(validate_url("https://vashti.chat", false).is_ok());
        assert!(validate_url("http://127.0.0.1:7781", false).is_ok());
        assert!(validate_url("http://localhost:7781", false).is_ok());
        assert!(validate_url("http://vashti.chat", false).is_err());
        assert!(validate_url("http://192.168.1.4:7781", false).is_err());
        assert!(validate_url("http://127.0.0.1:7771/api/version", true).is_ok());
        assert!(validate_url("http://192.168.1.4:7771/api/version", true).is_ok());
        assert!(validate_url("https://vashti.chat/releases", false).is_err());
        assert!(validate_url("https://user@vashti.chat", false).is_err());
    }

    #[test]
    fn manifest_must_match_every_constrained_request_field() {
        let stable = fixture_manifest("stable", "released");
        assert!(validate_manifest(&stable, &fixture_request("stable"), "linux-x86_64").is_ok());

        let mut wrong_channel = fixture_manifest("prerelease", "prerelease");
        assert!(
            validate_manifest(&wrong_channel, &fixture_request("stable"), "linux-x86_64").is_err()
        );
        wrong_channel.channel = "stable".to_string();
        assert!(
            validate_manifest(&wrong_channel, &fixture_request("stable"), "linux-x86_64").is_err()
        );

        let mut wrong_path = fixture_manifest("stable", "released");
        wrong_path.download_url = "/releases/v1.2.3/other.tar.gz".to_string();
        assert!(
            validate_manifest(&wrong_path, &fixture_request("stable"), "linux-x86_64").is_err()
        );

        let mut altered = fixture_manifest("stable", "released");
        altered.artifact.sha256 = "b".repeat(64);
        assert!(validate_manifest(&altered, &fixture_request("stable"), "linux-x86_64").is_err());
    }

    #[test]
    fn archive_extraction_uses_only_the_exact_packaged_binary() {
        let directory = test_directory("archive");
        std::fs::create_dir_all(&directory).expect("test directory should be created");
        let archive_path = directory.join("release.tar.gz");
        let binary_path = directory.join("vashti");
        write_archive(
            &archive_path,
            &[
                ("vashti-v1.2.3-linux-x86_64/README.md", b"ignore me"),
                ("vashti-v1.2.3-linux-x86_64/vashti", b"verified binary"),
            ],
        );

        extract_binary(&archive_path, &binary_path).expect("valid archive should extract");
        assert_eq!(
            std::fs::read(&binary_path).expect("binary should exist"),
            b"verified binary"
        );
        std::fs::remove_dir_all(directory).expect("test directory should be removed");
    }

    #[test]
    fn archive_extraction_rejects_duplicate_binary_entries() {
        let directory = test_directory("duplicate");
        std::fs::create_dir_all(&directory).expect("test directory should be created");
        let archive_path = directory.join("release.tar.gz");
        let binary_path = directory.join("vashti");
        write_archive(
            &archive_path,
            &[
                ("vashti-v1.2.3-linux-x86_64/vashti", b"first"),
                ("vashti-v1.2.3-linux-x86_64/vashti", b"second"),
            ],
        );

        assert!(extract_binary(&archive_path, &binary_path).is_err());
        std::fs::remove_dir_all(directory).expect("test directory should be removed");
    }

    #[test]
    fn work_directory_rejects_symlinks() {
        let directory = test_directory("work-dir");
        let target = directory.join("target");
        let link = directory.join("work");
        std::fs::create_dir_all(&target).expect("test target should be created");
        symlink(&target, &link).expect("test symlink should be created");

        let error = ensure_secure_work_dir(&link)
            .expect_err("managed update work directory must not be a symlink");
        assert!(error.to_string().contains("must be a real directory"));
        std::fs::remove_dir_all(directory).expect("test directory should be removed");
    }

    #[test]
    fn update_request_reader_rejects_symlinks_and_oversized_files() {
        let directory = test_directory("request-file");
        std::fs::create_dir_all(&directory).expect("test directory should be created");
        let valid_path = directory.join("valid.json");
        let valid_bytes = serde_json::to_vec(&fixture_request("stable"))
            .expect("fixture request should serialize");
        std::fs::write(&valid_path, &valid_bytes).expect("valid request should be written");
        assert_eq!(
            read_bounded_regular_file(&valid_path).expect("regular request should be read"),
            valid_bytes
        );

        let symlink_path = directory.join("request-link.json");
        symlink(&valid_path, &symlink_path).expect("test symlink should be created");
        assert!(read_bounded_regular_file(&symlink_path).is_err());

        let oversized_path = directory.join("oversized.json");
        std::fs::write(&oversized_path, vec![b'x'; MAX_REQUEST_BYTES as usize + 1])
            .expect("oversized request should be written");
        assert!(read_bounded_regular_file(&oversized_path).is_err());

        std::fs::remove_dir_all(directory).expect("test directory should be removed");
    }

    #[tokio::test]
    async fn rollback_copy_preserves_file_ownership_and_mode() {
        let directory = test_directory("preserve-metadata");
        std::fs::create_dir_all(&directory).expect("test directory should be created");
        let source = directory.join("backup.db");
        let destination = directory.join("restored.db");
        std::fs::write(&source, b"database contents").expect("source should be written");
        std::fs::set_permissions(&source, std::fs::Permissions::from_mode(0o640))
            .expect("source mode should be set");
        let source_metadata = std::fs::metadata(&source).expect("source metadata should exist");

        atomic_copy_preserving_metadata(&source, &destination)
            .await
            .expect("rollback copy should succeed");

        let restored_metadata =
            std::fs::metadata(&destination).expect("restored metadata should exist");
        assert_eq!(restored_metadata.uid(), source_metadata.uid());
        assert_eq!(restored_metadata.gid(), source_metadata.gid());
        assert_eq!(restored_metadata.mode() & 0o7777, 0o640);
        assert_eq!(
            std::fs::read(&destination).expect("restored file should be readable"),
            b"database contents"
        );

        std::fs::remove_dir_all(directory).expect("test directory should be removed");
    }
}
