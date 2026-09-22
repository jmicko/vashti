//! Read-only discovery and review bundles. The systemd install command repeats
//! privileged discovery instead of executing a potentially stale serialized plan.
use crate::config::Config;
use crate::deployment::Artifact;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener},
    path::Path,
};

#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    pub schema_version: u32,
    pub ready_to_apply: bool,
    pub service: String,
    pub active_state: String,
    pub original_listen: SocketAddr,
    pub proposed_config: Config,
    pub original_loaded_model_limit: Option<u32>,
    pub loaded_model_limit_policy: &'static str,
    pub proposed_drop_in_path: String,
    pub proposed_drop_in: String,
    pub proposed_gateway_unit: String,
    pub rollback: &'static str,
    pub remaining_checks: Vec<&'static str>,
}

pub(crate) fn property(service: &str, property: &str) -> Result<String, &'static str> {
    crate::systemd::command(&[
        "show",
        "--no-pager",
        "--value",
        "--property",
        property,
        service,
    ])
}

pub fn inspect(service: &str, config: &Config) -> Result<Plan, &'static str> {
    if !service.ends_with(".service")
        || !service
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-@".contains(&b))
        || service.starts_with('-')
    {
        return Err("invalid_service_name");
    }
    if property(service, "LoadState")?.trim() != "loaded" {
        return Err("service_not_loaded");
    }
    // EnvironmentFile and wrapper-based settings need a richer adapter. Do not
    // pretend the unit's Environment alone is the process's effective config.
    if !property(service, "EnvironmentFiles")?.trim().is_empty() {
        return Err("environment_file_requires_manual_discovery");
    }
    let environment = parse_environment(&property(service, "Environment")?)?;
    let configured_listen = match environment.get("OLLAMA_HOST") {
        Some(value) => parse_listen(value)?,
        None => "127.0.0.1:11434".parse().unwrap(),
    };
    // Go may serve an IPv4 wildcard through a dual-stack IPv6 socket. Preserve
    // the actual listener; ownership is checked with root privileges at install.
    let original_listen = crate::systemd::observed_listener(configured_listen)?;
    let limit = environment
        .get("OLLAMA_MAX_LOADED_MODELS")
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|_| "loaded_model_limit_invalid")
        })
        .transpose()?;
    let port = free_port_after(original_listen.port(), config.control_listen.port())?;
    let mut proposed = config.clone();
    proposed.listen = original_listen;
    proposed.state_dir = "/var/lib/kuno".into();
    proposed.activation_file = Some(crate::systemd::READY.into());
    proposed.upstream = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    proposed.validate()?;
    let active = property(service, "ActiveState")?;
    let active = match active.trim() {
        "active" => "active",
        "inactive" => "inactive",
        "failed" => "failed",
        _ => "other",
    };
    Ok(Plan {
        schema_version: 1,
        ready_to_apply: false,
        service: service.into(),
        active_state: active.into(),
        original_listen,
        proposed_drop_in_path: format!("/etc/systemd/system/{service}.d/zzzz-kuno.conf"),
        proposed_drop_in: crate::systemd::ollama_drop_in(proposed.upstream),
        proposed_gateway_unit: gateway_unit(service),
        proposed_config: proposed,
        original_loaded_model_limit: limit,
        loaded_model_limit_policy: "preserve_until_resource_scheduler_is_operational",
        rollback: "Remove only the unchanged Kuno-owned drop-in; reload systemd and restore the recorded original service state. Preserve all pre-existing drop-ins.",
        remaining_checks: vec![
            "Verify the actual listener owner and process environment, including wrappers and competing overrides.",
            "Installation interrupts active connections; clients can reconnect after verified cutover.",
            "Persist a protected atomic transaction journal before any service edit.",
            "Bind and health-check both services, with rollback on any failure; the proposed port is only a probe.",
            "Install the unprivileged Kuno service and verify privacy protections and boot recovery.",
        ],
    })
}

pub(crate) fn gateway_unit(service: &str) -> String {
    format!(
        "[Unit]\nDescription=Kuno inference gateway\nAfter=network.target {service}\nWants={service}\n\n[Service]\nType=simple\nDynamicUser=yes\nStateDirectory=kuno\nStateDirectoryMode=0700\nLoadCredential=control-token:/etc/kuno/control-token\nEnvironment=KUNO_CONTROL_TOKEN_FILE=%d/control-token\nExecStart=/usr/local/lib/kuno/kuno serve --config /etc/kuno/config.json\nRestart=on-failure\nRestartSec=3\nTimeoutStopSec=40\nNoNewPrivileges=yes\nProtectSystem=strict\nProtectHome=yes\nPrivateTmp=yes\nLimitCORE=0\nMemorySwapMax=0\nUMask=0077\n\n[Install]\nWantedBy=multi-user.target\n"
    )
}

/// Generate reviewable installation inputs in a new private directory. This is
/// staging only: none of these files is copied to the host's service paths.
pub fn bundle(plan: &Plan, output: &Path, executable: &[u8]) -> Result<(), &'static str> {
    let mut plan = plan.clone();
    plan.proposed_config.state_dir = "/var/lib/kuno".into();
    let mut random = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut random))
        .map_err(|_| "credential_generation_failed")?;
    let token: String = random.iter().map(|b| format!("{b:02x}")).collect();
    let mut config = plan.proposed_config.clone();
    config.state_dir = "/var/lib/kuno".into();
    let artifacts = vec![
        Artifact {
            path: "etc/kuno/config.json".into(),
            bytes: serde_json::to_vec_pretty(&config)
                .map_err(|_| "configuration_encoding_failed")?,
            mode: 0o644,
        },
        Artifact {
            path: "etc/kuno/control-token".into(),
            bytes: token.into_bytes(),
            mode: 0o600,
        },
        Artifact {
            path: "etc/systemd/system/kuno.service".into(),
            bytes: plan.proposed_gateway_unit.as_bytes().to_vec(),
            mode: 0o644,
        },
        Artifact {
            path: plan
                .proposed_drop_in_path
                .strip_prefix('/')
                .ok_or("bundle_path_invalid")?
                .into(),
            bytes: plan.proposed_drop_in.as_bytes().to_vec(),
            mode: 0o644,
        },
        Artifact {
            path: "usr/local/lib/kuno/kuno".into(),
            bytes: executable.to_vec(),
            mode: 0o755,
        },
    ];
    // All paths are validated before the first write, even for programmatic callers.
    if artifacts.iter().any(|a| {
        a.path
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    }) {
        return Err("bundle_path_invalid");
    }
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(output)
        .map_err(|_| "bundle_directory_must_be_new")?;
    for artifact in artifacts {
        let path = output.join(artifact.path);
        std::fs::create_dir_all(path.parent().ok_or("bundle_path_invalid")?)
            .map_err(|_| "bundle_write_failed")?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(artifact.mode).custom_flags(libc::O_NOFOLLOW);
        }
        let mut file = options.open(path).map_err(|_| "bundle_write_failed")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(std::fs::Permissions::from_mode(artifact.mode))
                .map_err(|_| "bundle_write_failed")?;
        }
        file.write_all(&artifact.bytes)
            .and_then(|_| file.sync_all())
            .map_err(|_| "bundle_write_failed")?;
    }
    crate::storage::write_json(&output.join("plan.json"), &plan)
        .map_err(|_| "bundle_write_failed")?;
    Ok(())
}

pub(crate) fn parse_listen(value: &str) -> Result<SocketAddr, &'static str> {
    let value = value.strip_prefix("http://").unwrap_or(value);
    value
        .parse()
        .map_err(|_| "ollama_host_requires_manual_discovery")
}

fn free_port_after(original: u16, control: u16) -> Result<u16, &'static str> {
    for port in original.saturating_add(1)..=original.saturating_add(100) {
        if port == original || port == control {
            continue;
        }
        if TcpListener::bind((Ipv4Addr::LOCALHOST, port)).is_ok() {
            return Ok(port);
        }
    }
    Err("no_private_port_available")
}

/// systemctl's quoted Environment rendering, not a shell. Reject unsupported
/// escapes rather than infer effective values incorrectly. Other variables are
/// transient and never appear in the plan or errors.
pub(crate) fn parse_environment(input: &str) -> Result<BTreeMap<String, String>, &'static str> {
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut quoted = false;
    let mut chars = input.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => quoted = !quoted,
            '\\' => match chars.next() {
                Some(c @ ('\\' | '"' | ' ')) => token.push(c),
                _ => return Err("environment_encoding_requires_manual_discovery"),
            },
            c if c.is_whitespace() && !quoted => {
                if !token.is_empty() {
                    tokens.push(std::mem::take(&mut token));
                }
            }
            c => token.push(c),
        }
    }
    if quoted {
        return Err("environment_encoding_requires_manual_discovery");
    }
    if !token.is_empty() {
        tokens.push(token);
    }
    let mut result = BTreeMap::new();
    for token in tokens {
        let (key, value) = token
            .split_once('=')
            .ok_or("environment_encoding_requires_manual_discovery")?;
        if matches!(key, "OLLAMA_HOST" | "OLLAMA_MAX_LOADED_MODELS") {
            result.insert(key.into(), value.into());
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_only_retains_allowlisted_settings() {
        let env = parse_environment(
            "OLLAMA_HOST=0.0.0.0:11434 \"SECRET=private value\" OLLAMA_MAX_LOADED_MODELS=1",
        )
        .unwrap();
        assert_eq!(env.len(), 2);
        assert_eq!(env["OLLAMA_MAX_LOADED_MODELS"], "1");
        assert_eq!(parse_listen(&env["OLLAMA_HOST"]).unwrap().port(), 11434);
    }

    #[test]
    fn unsupported_host_and_encoding_fail_without_echoing_input() {
        assert_eq!(
            parse_listen("http://secret@host/path").unwrap_err(),
            "ollama_host_requires_manual_discovery"
        );
        assert!(parse_environment("\"unclosed").is_err());
        assert!(parse_environment("VALUE=\\x41").is_err());
        assert!(parse_listen("localhost:11434").is_err());
    }
}
