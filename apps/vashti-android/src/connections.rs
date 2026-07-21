use std::{
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures_util::StreamExt;
use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{secrets::SessionStore, validation::normalize_base_url};

pub const SUPPORTED_API_VERSION: u32 = 1;
const MAX_VERSION_RESPONSE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub instance_id: String,
    pub api_version: u32,
    pub allow_insecure_http: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConnectionInput {
    pub name: String,
    pub base_url: String,
    #[serde(default)]
    pub allow_insecure_http: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectionConfig {
    pub active_connection_id: Option<String>,
    pub connections: Vec<Connection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectionSnapshot {
    pub active_connection_id: Option<String>,
    pub connections: Vec<Connection>,
}

#[derive(Debug, Deserialize)]
struct VersionResponse {
    name: String,
    instance_id: String,
    api_version: u32,
}

pub struct NativeState {
    pub config_path: PathBuf,
    pub config: RwLock<ConnectionConfig>,
    pub client: reqwest::Client,
    pub sessions: SessionStore,
    pub active_requests: Mutex<std::collections::HashMap<String, CancellationToken>>,
}

impl NativeState {
    pub fn load(app: &AppHandle) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_dir = app.path().app_config_dir()?;
        std::fs::create_dir_all(&config_dir)?;
        let config_path = config_dir.join("connections.json");
        let config = match std::fs::read(&config_path) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(config) => config,
                Err(error) => {
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis();
                    let backup_path = config_dir.join(format!("connections-{timestamp}.corrupt"));
                    let _ = std::fs::rename(&config_path, &backup_path);
                    eprintln!(
                        "saved server configuration was invalid and was moved to {}: {error}",
                        backup_path.display()
                    );
                    ConnectionConfig::default()
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                ConnectionConfig::default()
            }
            Err(error) => return Err(error.into()),
        };
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .redirect(Policy::none())
            .user_agent(format!("Vashti-Android/{}", env!("CARGO_PKG_VERSION")))
            .build()?;

        Ok(Self {
            config_path,
            config: RwLock::new(config),
            client,
            sessions: SessionStore::new(),
            active_requests: Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub async fn active_connection(&self) -> Result<Connection, String> {
        let config = self.config.read().await;
        config
            .active_connection_id
            .as_ref()
            .and_then(|active_id| {
                config
                    .connections
                    .iter()
                    .find(|connection| &connection.id == active_id)
            })
            .cloned()
            .ok_or_else(|| "No Vashti server is selected".to_string())
    }

    async fn snapshot(&self) -> ConnectionSnapshot {
        let config = self.config.read().await;
        ConnectionSnapshot {
            active_connection_id: config.active_connection_id.clone(),
            connections: config.connections.clone(),
        }
    }

    async fn save_config(&self, config: &ConnectionConfig) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(config).map_err(|error| error.to_string())?;
        let temp_path = self.config_path.with_extension("json.tmp");
        tokio::fs::write(&temp_path, bytes)
            .await
            .map_err(|error| error.to_string())?;
        tokio::fs::rename(&temp_path, &self.config_path)
            .await
            .map_err(|error| error.to_string())
    }

    async fn validate_connection(&self, input: &ConnectionInput) -> Result<Connection, String> {
        let name = input.name.trim();
        if name.is_empty() || name.chars().count() > 80 {
            return Err("Server name must be between 1 and 80 characters".to_string());
        }
        let base_url = normalize_base_url(&input.base_url, input.allow_insecure_http)?;
        let version_url = base_url
            .join("api/version")
            .map_err(|_| "Server URL is invalid".to_string())?;
        let response = self
            .client
            .get(version_url)
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .map_err(connection_error)?;
        if !response.status().is_success() {
            return Err(format!(
                "The server returned {} while checking compatibility",
                response.status()
            ));
        }
        let version: VersionResponse = serde_json::from_slice(
            &response_bytes_limited(response, MAX_VERSION_RESPONSE_BYTES).await?,
        )
        .map_err(|_| "That address did not return Vashti server metadata".to_string())?;
        if version.name != "vashti" {
            return Err("That address is not a Vashti server".to_string());
        }
        if version.api_version != SUPPORTED_API_VERSION {
            return Err(format!(
                "This app supports Vashti API version {}, but the server uses version {}",
                SUPPORTED_API_VERSION, version.api_version
            ));
        }
        let instance_id = Uuid::parse_str(version.instance_id.trim())
            .map_err(|_| "The server returned an invalid identity".to_string())?;
        if instance_id.is_nil() {
            return Err("The server returned an invalid identity".to_string());
        }

        Ok(Connection {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            base_url: base_url.as_str().trim_end_matches('/').to_string(),
            instance_id: instance_id.to_string(),
            api_version: version.api_version,
            allow_insecure_http: input.allow_insecure_http,
        })
    }
}

#[tauri::command]
pub async fn native_list_connections(
    state: State<'_, NativeState>,
) -> Result<ConnectionSnapshot, String> {
    Ok(state.snapshot().await)
}

#[tauri::command]
pub async fn native_add_connection(
    state: State<'_, NativeState>,
    input: ConnectionInput,
) -> Result<ConnectionSnapshot, String> {
    let connection = state.validate_connection(&input).await?;
    let mut config = state.config.write().await;
    if config
        .connections
        .iter()
        .any(|existing| existing.instance_id == connection.instance_id)
    {
        return Err("This Vashti server is already saved".to_string());
    }
    config.active_connection_id = Some(connection.id.clone());
    config.connections.push(connection);
    state.save_config(&config).await?;
    Ok(ConnectionSnapshot {
        active_connection_id: config.active_connection_id.clone(),
        connections: config.connections.clone(),
    })
}

#[tauri::command]
pub async fn native_update_connection(
    state: State<'_, NativeState>,
    id: String,
    input: ConnectionInput,
) -> Result<ConnectionSnapshot, String> {
    let mut replacement = state.validate_connection(&input).await?;
    replacement.id.clone_from(&id);
    let mut config = state.config.write().await;
    if config
        .connections
        .iter()
        .any(|existing| existing.id != id && existing.instance_id == replacement.instance_id)
    {
        return Err("This Vashti server is already saved".to_string());
    }
    let position = config
        .connections
        .iter()
        .position(|connection| connection.id == id)
        .ok_or_else(|| "Saved server was not found".to_string())?;
    let previous = &config.connections[position];
    let transport_changed = previous.instance_id != replacement.instance_id
        || previous.base_url != replacement.base_url
        || previous.allow_insecure_http != replacement.allow_insecure_http;
    config.connections[position] = replacement;
    state.save_config(&config).await?;
    drop(config);
    if transport_changed {
        state.sessions.delete(&id).await?;
    }
    Ok(state.snapshot().await)
}

#[tauri::command]
pub async fn native_remove_connection(
    state: State<'_, NativeState>,
    id: String,
) -> Result<ConnectionSnapshot, String> {
    let mut config = state.config.write().await;
    let previous_len = config.connections.len();
    config.connections.retain(|connection| connection.id != id);
    if config.connections.len() == previous_len {
        return Err("Saved server was not found".to_string());
    }
    if config.active_connection_id.as_deref() == Some(&id) {
        config.active_connection_id = config.connections.first().map(|item| item.id.clone());
    }
    state.save_config(&config).await?;
    drop(config);
    state.sessions.delete(&id).await?;
    Ok(state.snapshot().await)
}

#[tauri::command]
pub async fn native_select_connection(
    state: State<'_, NativeState>,
    id: String,
) -> Result<ConnectionSnapshot, String> {
    let mut config = state.config.write().await;
    if !config
        .connections
        .iter()
        .any(|connection| connection.id == id)
    {
        return Err("Saved server was not found".to_string());
    }
    config.active_connection_id = Some(id);
    state.save_config(&config).await?;
    Ok(ConnectionSnapshot {
        active_connection_id: config.active_connection_id.clone(),
        connections: config.connections.clone(),
    })
}

#[tauri::command]
pub async fn native_sync_active_identity(
    state: State<'_, NativeState>,
    instance_id: String,
    api_version: u32,
) -> Result<bool, String> {
    if api_version != SUPPORTED_API_VERSION {
        return Err(format!(
            "This app supports Vashti API version {}, but the server uses version {}",
            SUPPORTED_API_VERSION, api_version
        ));
    }
    let instance_id = Uuid::parse_str(instance_id.trim())
        .map_err(|_| "The server returned an invalid identity".to_string())?;
    if instance_id.is_nil() {
        return Err("The server returned an invalid identity".to_string());
    }
    let instance_id = instance_id.to_string();

    let mut config = state.config.write().await;
    let active_id = config
        .active_connection_id
        .clone()
        .ok_or_else(|| "No Vashti server is selected".to_string())?;
    let connection = config
        .connections
        .iter_mut()
        .find(|connection| connection.id == active_id)
        .ok_or_else(|| "The selected Vashti server was not found".to_string())?;
    if connection.instance_id == instance_id && connection.api_version == api_version {
        return Ok(false);
    }

    connection.instance_id = instance_id;
    connection.api_version = api_version;
    state.save_config(&config).await?;
    drop(config);
    state.sessions.delete(&active_id).await?;
    Ok(true)
}

fn connection_error(error: reqwest::Error) -> String {
    if error.is_timeout() {
        "Timed out while connecting to the Vashti server".to_string()
    } else if error.is_connect() {
        "Could not connect to that server. Check the address and network.".to_string()
    } else {
        format!("Could not verify the Vashti server: {error}")
    }
}

async fn response_bytes_limited(
    response: reqwest::Response,
    maximum: usize,
) -> Result<Vec<u8>, String> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err("The server metadata response was too large".to_string());
    }

    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(connection_error)?;
        if bytes.len().saturating_add(chunk.len()) > maximum {
            return Err("The server metadata response was too large".to_string());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}
