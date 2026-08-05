use std::{
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct Config {
    pub app_root: PathBuf,
    pub data_dir: PathBuf,
    pub database_path: PathBuf,
    pub bind_addr: SocketAddr,
    pub session_cookie_name: String,
    pub session_ttl_seconds: i64,
    pub managed_updates: bool,
    pub update_base_url: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to resolve current directory: {0}")]
    CurrentDir(std::io::Error),
    #[error("invalid VASHTI_BIND address: {0}")]
    BindAddr(#[from] std::net::AddrParseError),
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let app_root = env::current_dir().map_err(ConfigError::CurrentDir)?;
        let data_dir = env::var_os("VASHTI_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| default_data_dir(&app_root));
        let data_dir = if data_dir.is_absolute() {
            data_dir
        } else {
            app_root.join(data_dir)
        };

        let bind_addr = env::var("VASHTI_BIND")
            .unwrap_or_else(|_| "0.0.0.0:7771".to_string())
            .parse()?;

        Ok(Self {
            app_root,
            database_path: data_dir.join("app.db"),
            data_dir,
            bind_addr,
            session_cookie_name: "vashti_session".to_string(),
            session_ttl_seconds: 60 * 60 * 24 * 30,
            managed_updates: env_flag("VASHTI_MANAGED_UPDATES"),
            update_base_url: env::var("VASHTI_UPDATE_BASE_URL")
                .unwrap_or_else(|_| "https://vashti.chat".to_string()),
        })
    }

    pub fn uploads_dir(&self) -> PathBuf {
        self.data_dir.join("uploads")
    }

    pub fn persona_avatars_dir(&self) -> PathBuf {
        self.data_dir.join("persona-avatars")
    }

    pub fn tmp_dir(&self) -> PathBuf {
        self.data_dir.join("tmp")
    }

    pub fn update_dir(&self) -> PathBuf {
        self.data_dir.join("update")
    }

    pub fn update_request_path(&self) -> PathBuf {
        self.update_dir().join("request.json")
    }

    pub fn update_processing_request_path(&self) -> PathBuf {
        self.update_dir().join("request.in-progress.json")
    }

    pub fn update_status_path(&self) -> PathBuf {
        self.update_dir().join("status.json")
    }
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
}

fn default_data_dir(app_root: &Path) -> PathBuf {
    if app_root.join("apps/vashti").is_dir() {
        PathBuf::from("apps/vashti/data")
    } else {
        PathBuf::from("data")
    }
}
