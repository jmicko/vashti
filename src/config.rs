use std::{env, net::SocketAddr, path::PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub data_dir: PathBuf,
    pub database_path: PathBuf,
    pub bind_addr: SocketAddr,
    pub session_cookie_name: String,
    pub session_ttl_seconds: i64,
    pub cookie_secure: bool,
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
        let data_dir = env::var_os("VASHTI_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data"));
        let data_dir = if data_dir.is_absolute() {
            data_dir
        } else {
            env::current_dir()
                .map_err(ConfigError::CurrentDir)?
                .join(data_dir)
        };

        let bind_addr = env::var("VASHTI_BIND")
            .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
            .parse()?;

        let cookie_secure = env::var("VASHTI_COOKIE_SECURE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);

        Ok(Self {
            database_path: data_dir.join("app.db"),
            data_dir,
            bind_addr,
            session_cookie_name: "vashti_session".to_string(),
            session_ttl_seconds: 60 * 60 * 24 * 30,
            cookie_secure,
        })
    }

    pub fn uploads_dir(&self) -> PathBuf {
        self.data_dir.join("uploads")
    }

    pub fn tmp_dir(&self) -> PathBuf {
        self.data_dir.join("tmp")
    }
}
