use std::{path::Path, time::Duration};

use futures_util::StreamExt;
use reqwest::StatusCode;
use semver::Version;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;
use vashti_update_manifest::{SCHEMA_VERSION, SignedArtifact, verify_release_signature};

use crate::{auth::service::unix_timestamp, config::Config, settings};

const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_MANIFEST_BYTES: usize = 256 * 1024;
const UPDATE_CHECK_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Default)]
struct UpdateCheckState {
    available: Option<AvailableUpdate>,
    checked_at: Option<i64>,
    check_error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UpdateStatusResponse {
    pub managed_updates: bool,
    pub target: Option<&'static str>,
    pub channel: String,
    pub current_version: String,
    pub available: Option<AvailableUpdate>,
    pub checked_at: Option<i64>,
    pub check_error: Option<String>,
    pub operation: UpdateOperationStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AvailableUpdate {
    pub version: String,
    pub notes: Option<String>,
    pub release_status: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UpdateOperationStatus {
    pub state: String,
    pub version: Option<String>,
    pub message: Option<String>,
    pub updated_at: Option<i64>,
}

impl Default for UpdateOperationStatus {
    fn default() -> Self {
        Self {
            state: "idle".to_string(),
            version: None,
            message: None,
            updated_at: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct RemoteUpdateManifest {
    schema_version: u32,
    channel: String,
    release_status: String,
    notes: Option<String>,
    artifact: SignedArtifact,
    signature: String,
    download_url: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateRequest {
    pub schema_version: u32,
    pub channel: String,
    pub version: String,
    pub requested_at: i64,
}

#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    #[error("database operation failed: {0}")]
    Database(#[from] sqlx::Error),
    #[error("update service request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("update service URL is invalid")]
    InvalidBaseUrl,
    #[error("update manifest is invalid: {0}")]
    InvalidManifest(String),
    #[error("managed updates are not installed for this Vashti instance")]
    ManagedUpdatesUnavailable,
    #[error("no newer update is available")]
    NoUpdateAvailable,
    #[error("an update request is already pending")]
    RequestPending,
    #[error("failed to write update state: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to encode update state: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct UpdateManager {
    state: RwLock<UpdateCheckState>,
    install_lock: Mutex<()>,
}

impl UpdateManager {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(UpdateCheckState::default()),
            install_lock: Mutex::new(()),
        }
    }

    pub async fn status(
        &self,
        pool: &SqlitePool,
        config: &Config,
    ) -> Result<UpdateStatusResponse, UpdateError> {
        let channel = settings::service::get_update_channel(pool).await?;
        let state = self.state.read().await.clone();
        let operation = read_operation_status(&config.update_status_path()).await;

        Ok(UpdateStatusResponse {
            managed_updates: config.managed_updates,
            target: update_target(),
            channel,
            current_version: version_label(CURRENT_VERSION),
            available: state.available,
            checked_at: state.checked_at,
            check_error: state.check_error,
            operation,
        })
    }

    pub async fn check_for_update(
        &self,
        pool: &SqlitePool,
        client: &reqwest::Client,
        config: &Config,
    ) -> Result<UpdateStatusResponse, UpdateError> {
        let channel = settings::service::get_update_channel(pool).await?;
        let checked_at = unix_timestamp();
        let result = fetch_update(client, config, &channel).await;

        {
            let mut state = self.state.write().await;
            state.checked_at = Some(checked_at);
            match &result {
                Ok(available) => {
                    state.available.clone_from(available);
                    state.check_error = None;
                }
                Err(error) => {
                    state.available = None;
                    state.check_error = Some(error.to_string());
                }
            }
        }

        result?;
        self.status(pool, config).await
    }

    pub async fn request_install(
        &self,
        pool: &SqlitePool,
        config: &Config,
    ) -> Result<UpdateStatusResponse, UpdateError> {
        let _install_guard = self.install_lock.lock().await;
        if !config.managed_updates {
            return Err(UpdateError::ManagedUpdatesUnavailable);
        }
        if tokio::fs::try_exists(config.update_request_path()).await?
            || tokio::fs::try_exists(config.update_processing_request_path()).await?
        {
            return Err(UpdateError::RequestPending);
        }

        let channel = settings::service::get_update_channel(pool).await?;
        let available = self
            .state
            .read()
            .await
            .available
            .clone()
            .ok_or(UpdateError::NoUpdateAvailable)?;
        ensure_newer_version(&available.version)?;

        tokio::fs::create_dir_all(config.update_dir()).await?;
        let now = unix_timestamp();
        let status = UpdateOperationStatus {
            state: "requested".to_string(),
            version: Some(available.version.clone()),
            message: Some("Update request accepted.".to_string()),
            updated_at: Some(now),
        };
        write_json_atomic(&config.update_status_path(), &status).await?;
        let request_result = write_json_atomic(
            &config.update_request_path(),
            &UpdateRequest {
                schema_version: SCHEMA_VERSION,
                channel,
                version: available.version.clone(),
                requested_at: now,
            },
        )
        .await;
        if let Err(error) = request_result {
            // A directory-sync error can arrive after the watched file was renamed,
            // and systemd may already have claimed it. Do not overwrite worker state
            // or tell the admin the handoff failed once either request file exists.
            let handed_off = tokio::fs::try_exists(config.update_request_path())
                .await
                .unwrap_or(false)
                || tokio::fs::try_exists(config.update_processing_request_path())
                    .await
                    .unwrap_or(false);
            if handed_off {
                tracing::warn!(
                    ?error,
                    "update request was committed but its directory sync failed"
                );
            } else {
                let failed_status = UpdateOperationStatus {
                    state: "failed".to_string(),
                    version: Some(available.version),
                    message: Some(
                        "The update request could not be handed to the updater.".to_string(),
                    ),
                    updated_at: Some(unix_timestamp()),
                };
                if let Err(status_error) =
                    write_json_atomic(&config.update_status_path(), &failed_status).await
                {
                    tracing::error!(
                        ?status_error,
                        "failed to record a rejected managed update request"
                    );
                }
                return Err(error);
            }
        }

        self.status(pool, config).await
    }
}

async fn fetch_update(
    client: &reqwest::Client,
    config: &Config,
    channel: &str,
) -> Result<Option<AvailableUpdate>, UpdateError> {
    let target = update_target().ok_or_else(|| {
        UpdateError::InvalidManifest("this operating system or architecture is unsupported".into())
    })?;
    let base_url = parse_update_base_url(&config.update_base_url)?;
    let endpoint = base_url
        .join(&format!("/api/updates/{channel}/{target}"))
        .map_err(|_| UpdateError::InvalidBaseUrl)?;
    let response = client
        .get(endpoint)
        .timeout(UPDATE_CHECK_TIMEOUT)
        .send()
        .await?;
    if response.status() == StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let response = response.error_for_status()?;
    let manifest = read_remote_manifest(response).await?;
    validate_remote_manifest(&manifest, channel, target)?;

    if ensure_newer_version(&manifest.artifact.version).is_err() {
        return Ok(None);
    }

    Ok(Some(AvailableUpdate {
        version: manifest.artifact.version,
        notes: manifest.notes,
        release_status: manifest.release_status,
    }))
}

async fn read_remote_manifest(
    response: reqwest::Response,
) -> Result<RemoteUpdateManifest, UpdateError> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_MANIFEST_BYTES as u64)
    {
        return Err(UpdateError::InvalidManifest(
            "update manifest exceeds the size limit".to_string(),
        ));
    }

    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if bytes.len().saturating_add(chunk.len()) > MAX_MANIFEST_BYTES {
            return Err(UpdateError::InvalidManifest(
                "update manifest exceeds the size limit".to_string(),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }

    serde_json::from_slice(&bytes).map_err(UpdateError::from)
}

fn parse_update_base_url(value: &str) -> Result<reqwest::Url, UpdateError> {
    let url = reqwest::Url::parse(value).map_err(|_| UpdateError::InvalidBaseUrl)?;
    let local_http = url.scheme() == "http"
        && url
            .host_str()
            .is_some_and(|host| matches!(host, "localhost" | "127.0.0.1" | "::1"));
    if url.host_str().is_none()
        || (!local_http && url.scheme() != "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(UpdateError::InvalidBaseUrl);
    }
    Ok(url)
}

fn validate_remote_manifest(
    manifest: &RemoteUpdateManifest,
    channel: &str,
    target: &str,
) -> Result<(), UpdateError> {
    if manifest.schema_version != SCHEMA_VERSION {
        return Err(UpdateError::InvalidManifest(
            "unsupported manifest schema".to_string(),
        ));
    }
    if manifest.channel != channel || manifest.artifact.target != target {
        return Err(UpdateError::InvalidManifest(
            "channel or target does not match the request".to_string(),
        ));
    }
    let expected_status = match channel {
        "stable" => "released",
        "prerelease" => manifest.release_status.as_str(),
        _ => {
            return Err(UpdateError::InvalidManifest(
                "unsupported update channel".to_string(),
            ));
        }
    };
    if manifest.release_status != expected_status
        || !matches!(manifest.release_status.as_str(), "released" | "prerelease")
    {
        return Err(UpdateError::InvalidManifest(
            "release status does not match the selected channel".to_string(),
        ));
    }
    let expected_download_url = format!(
        "/releases/{}/{}",
        manifest.artifact.version, manifest.artifact.filename
    );
    if manifest.download_url != expected_download_url {
        return Err(UpdateError::InvalidManifest(
            "artifact URL does not match signed metadata".to_string(),
        ));
    }
    verify_release_signature(&manifest.artifact, &manifest.signature)
        .map_err(|error| UpdateError::InvalidManifest(error.to_string()))
}

fn ensure_newer_version(candidate: &str) -> Result<(), UpdateError> {
    let candidate = parse_version(candidate)?;
    let current = parse_version(CURRENT_VERSION)?;
    if candidate <= current {
        return Err(UpdateError::NoUpdateAvailable);
    }
    Ok(())
}

pub fn parse_version(value: &str) -> Result<Version, UpdateError> {
    Version::parse(value.trim_start_matches('v'))
        .map_err(|_| UpdateError::InvalidManifest("release version is invalid".to_string()))
}

pub fn version_label(value: &str) -> String {
    format!("v{}", value.trim_start_matches('v'))
}

pub const fn update_target() -> Option<&'static str> {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("linux-x86_64")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Some("linux-aarch64")
    } else {
        None
    }
}

pub async fn read_operation_status(path: &Path) -> UpdateOperationStatus {
    let Ok(bytes) = tokio::fs::read(path).await else {
        return UpdateOperationStatus::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

pub async fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), UpdateError> {
    let parent = path.parent().ok_or_else(|| {
        UpdateError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "update path has no parent directory",
        ))
    })?;
    tokio::fs::create_dir_all(parent).await?;
    let temporary_path = parent.join(format!(".update-{}.tmp", Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(value)?;
    let write_result: Result<(), std::io::Error> = async {
        let mut temporary = tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o660)
            .open(&temporary_path)
            .await?;
        temporary.write_all(&bytes).await?;
        temporary.flush().await?;
        temporary.sync_all().await?;
        drop(temporary);
        tokio::fs::rename(&temporary_path, path).await?;
        Ok(())
    }
    .await;
    if write_result.is_err() {
        let _ = tokio::fs::remove_file(&temporary_path).await;
    }
    write_result?;
    if let Err(error) = sync_directory(parent).await {
        tracing::warn!(
            ?error,
            ?path,
            "atomic update state was committed but its directory did not sync"
        );
    }
    Ok(())
}

async fn sync_directory(path: &Path) -> Result<(), std::io::Error> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || std::fs::File::open(path)?.sync_all())
        .await
        .map_err(std::io::Error::other)?
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_SIGNATURE: &str =
        "hQJFKgrhovM2Sa9XtxaNgftzQosjzge6uUPDZBYf97/sIYabYPVqilGNc+L6K/dhB92A+dhhTPvCztd13NO9CQ==";

    fn fixture_manifest(channel: &str, release_status: &str) -> RemoteUpdateManifest {
        RemoteUpdateManifest {
            schema_version: SCHEMA_VERSION,
            channel: channel.to_string(),
            release_status: release_status.to_string(),
            notes: None,
            artifact: SignedArtifact {
                version: "v1.2.3".to_string(),
                target: "linux-x86_64".to_string(),
                filename: "vashti-linux-x86_64.tar.gz".to_string(),
                sha256: "a".repeat(64),
                size_bytes: 42,
            },
            signature: FIXTURE_SIGNATURE.to_string(),
            download_url: "/releases/v1.2.3/vashti-linux-x86_64.tar.gz".to_string(),
        }
    }

    #[test]
    fn version_comparison_rejects_current_and_older_releases() {
        assert!(ensure_newer_version(CURRENT_VERSION).is_err());
        assert!(ensure_newer_version("v0.0.1").is_err());
        assert!(ensure_newer_version("v999.0.0").is_ok());
    }

    #[test]
    fn update_request_rejects_unknown_fields() {
        let request = r#"{
            "schema_version": 1,
            "channel": "stable",
            "version": "v1.0.0",
            "requested_at": 1,
            "command": "anything"
        }"#;
        assert!(serde_json::from_str::<UpdateRequest>(request).is_err());
    }

    #[test]
    fn update_base_url_requires_a_trusted_origin_shape() {
        assert!(parse_update_base_url("https://vashti.chat").is_ok());
        assert!(parse_update_base_url("http://127.0.0.1:7781").is_ok());
        assert!(parse_update_base_url("http://vashti.chat").is_err());
        assert!(parse_update_base_url("https://vashti.chat/releases").is_err());
        assert!(parse_update_base_url("https://user@vashti.chat").is_err());
    }

    #[test]
    fn remote_manifest_rejects_channel_status_path_and_signature_changes() {
        assert!(
            validate_remote_manifest(
                &fixture_manifest("stable", "released"),
                "stable",
                "linux-x86_64"
            )
            .is_ok()
        );
        assert!(
            validate_remote_manifest(
                &fixture_manifest("stable", "prerelease"),
                "stable",
                "linux-x86_64"
            )
            .is_err()
        );
        assert!(
            validate_remote_manifest(
                &fixture_manifest("prerelease", "prerelease"),
                "prerelease",
                "linux-x86_64"
            )
            .is_ok()
        );

        let mut wrong_path = fixture_manifest("stable", "released");
        wrong_path.download_url = "/releases/v1.2.3/other.tar.gz".to_string();
        assert!(validate_remote_manifest(&wrong_path, "stable", "linux-x86_64").is_err());

        let mut altered = fixture_manifest("stable", "released");
        altered.artifact.size_bytes += 1;
        assert!(validate_remote_manifest(&altered, "stable", "linux-x86_64").is_err());
    }

    #[tokio::test]
    async fn atomic_update_state_replaces_complete_json_without_temp_files() {
        let directory =
            std::env::temp_dir().join(format!("vashti-update-state-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&directory)
            .await
            .expect("test directory should be created");
        let path = directory.join("status.json");
        let initial = UpdateOperationStatus {
            state: "requested".to_string(),
            version: Some("v1.2.3".to_string()),
            message: None,
            updated_at: Some(1),
        };
        let replacement = UpdateOperationStatus {
            state: "succeeded".to_string(),
            version: Some("v1.2.3".to_string()),
            message: Some("complete".to_string()),
            updated_at: Some(2),
        };

        write_json_atomic(&path, &initial)
            .await
            .expect("initial state should be written");
        write_json_atomic(&path, &replacement)
            .await
            .expect("replacement state should be written");

        let stored = serde_json::from_slice::<UpdateOperationStatus>(
            &tokio::fs::read(&path)
                .await
                .expect("state should be readable"),
        )
        .expect("state should contain complete JSON");
        assert_eq!(stored.state, "succeeded");
        assert_eq!(stored.updated_at, Some(2));
        let mut entries = tokio::fs::read_dir(&directory)
            .await
            .expect("test directory should be readable");
        let mut names = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .expect("test directory entry should be readable")
        {
            names.push(entry.file_name());
        }
        assert_eq!(names, vec![std::ffi::OsString::from("status.json")]);
        tokio::fs::remove_dir_all(directory)
            .await
            .expect("test directory should be removed");
    }
}
