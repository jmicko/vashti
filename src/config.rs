use std::{env, net::SocketAddr, path::PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub app_root: PathBuf,
    pub data_dir: PathBuf,
    pub database_path: PathBuf,
    pub bind_addr: SocketAddr,
    pub session_cookie_name: String,
    pub session_ttl_seconds: i64,
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
            .unwrap_or_else(|| PathBuf::from("data"));
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
        })
    }

    pub fn uploads_dir(&self) -> PathBuf {
        self.data_dir.join("uploads")
    }

    pub fn tmp_dir(&self) -> PathBuf {
        self.data_dir.join("tmp")
    }
}
