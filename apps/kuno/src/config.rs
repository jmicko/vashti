use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub listen: SocketAddr,
    pub control_listen: SocketAddr,
    /// A literal local address avoids DNS changes, proxy environment variables,
    /// remote redirects, and arbitrary URL/path forwarding.
    pub upstream: SocketAddr,
    pub state_dir: PathBuf,
    /// Managed installs remain paused until the installer activates them. On
    /// subsequent boots this root-owned marker permits ordinary admission.
    pub activation_file: Option<PathBuf>,
    pub max_waiting: usize,
    pub queue_timeout_seconds: u64,
    /// Zero disables the response-header deadline for long prompt processing.
    pub response_header_timeout_seconds: u64,
    /// Zero disables the response-body idle deadline for long inference.
    pub stream_idle_timeout_seconds: u64,
    pub managed: Option<crate::pool::PoolConfig>,
    pub image: Option<crate::image_runtime::ImageConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:11433".parse().unwrap(),
            control_listen: "127.0.0.1:11432".parse().unwrap(),
            upstream: "127.0.0.1:11434".parse().unwrap(),
            state_dir: PathBuf::from("apps/kuno/data"),
            activation_file: None,
            max_waiting: 32,
            queue_timeout_seconds: 120,
            response_header_timeout_seconds: 0,
            stream_idle_timeout_seconds: 0,
            managed: None,
            image: None,
        }
    }
}

impl Config {
    pub fn read(path: &Path) -> Result<Self, &'static str> {
        let bytes = std::fs::read(path).map_err(|_| "configuration_read_failed")?;
        let config: Self = serde_json::from_slice(&bytes).map_err(|_| "configuration_invalid")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.state_dir.as_os_str().is_empty() {
            return Err("state_directory_required");
        }
        if !self.upstream.ip().is_loopback() || !self.control_listen.ip().is_loopback() {
            return Err("upstream_and_control_must_be_loopback");
        }
        // Conservatively reject same-port setups even across address families.
        let ports = [
            self.listen.port(),
            self.control_listen.port(),
            self.upstream.port(),
        ];
        if ports.contains(&0)
            || ports[0] == ports[1]
            || ports[0] == ports[2]
            || ports[1] == ports[2]
        {
            return Err("ports_must_be_nonzero_and_distinct");
        }
        if let Some(managed) = &self.managed {
            managed.validate(&ports)?;
        }
        if let Some(image) = &self.image {
            image.validate()?;
            if self.managed.is_none() {
                return Err("image_requires_managed_pool");
            }
        }
        if self.max_waiting > 1024
            || self.queue_timeout_seconds == 0
            || [
                self.queue_timeout_seconds,
                self.response_header_timeout_seconds,
                self.stream_idle_timeout_seconds,
            ]
            .iter()
            .any(|&n| n > 86400)
        {
            return Err("invalid_admission_bounds");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_deadlines_default_to_disabled_but_explicit_values_survive_loading() {
        let new: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(new.response_header_timeout_seconds, 0);
        assert_eq!(new.stream_idle_timeout_seconds, 0);
        new.validate().unwrap();
        let installed: Config = serde_json::from_str(
            r#"{"response_header_timeout_seconds":300,"stream_idle_timeout_seconds":600}"#,
        )
        .unwrap();
        installed.validate().unwrap();
        assert_eq!(installed.response_header_timeout_seconds, 300);
        assert_eq!(installed.stream_idle_timeout_seconds, 600);
    }

    #[test]
    fn only_inference_deadlines_allow_zero_and_all_timeouts_remain_bounded() {
        for (headers, idle) in [(0, 0), (0, 86400), (86400, 0), (86400, 86400)] {
            Config {
                response_header_timeout_seconds: headers,
                stream_idle_timeout_seconds: idle,
                ..Config::default()
            }
            .validate()
            .unwrap();
        }
        for config in [
            Config {
                queue_timeout_seconds: 0,
                ..Config::default()
            },
            Config {
                queue_timeout_seconds: 86401,
                ..Config::default()
            },
            Config {
                response_header_timeout_seconds: 86401,
                ..Config::default()
            },
            Config {
                stream_idle_timeout_seconds: 86401,
                ..Config::default()
            },
        ] {
            assert_eq!(config.validate(), Err("invalid_admission_bounds"));
        }
    }
}
