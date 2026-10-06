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
        proposed_gateway_unit: configured_gateway_unit(service, &proposed)?,
        proposed_config: proposed.clone(),
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
    // Render-node access supports native GPU telemetry without root or broad
    // capabilities. Do not reference a missing group on NVIDIA-only/minimal hosts.
    let render_access = std::fs::read_to_string("/etc/group")
        .is_ok_and(|groups| groups.lines().any(|line| line.starts_with("render:")));
    let gpu_access = if render_access {
        "SupplementaryGroups=render\n"
    } else {
        ""
    };
    format!(
        "[Unit]\nDescription=Kuno inference gateway\nAfter=network.target {service}\nWants={service}\n\n[Service]\nType=simple\nDynamicUser=yes\n{gpu_access}StateDirectory=kuno\nStateDirectoryMode=0700\nLoadCredential=control-token:/etc/kuno/control-token\nEnvironment=KUNO_CONTROL_TOKEN_FILE=%d/control-token\nExecStart=/usr/local/lib/kuno/kuno serve --config /etc/kuno/config.json\nRestart=on-failure\nRestartSec=3\nTimeoutStopSec=40\nNoNewPrivileges=yes\nProtectSystem=strict\nProtectHome=yes\nPrivateTmp=yes\nLimitCORE=0\nMemorySwapMax=0\nUMask=0077\n\n[Install]\nWantedBy=multi-user.target\n"
    )
}

/// Map the shared model store inside the service without widening host ACLs.
/// The daemon rewrites only the worker's runtime view to this fixed mount.
pub(crate) fn configured_gateway_unit(
    service: &str,
    config: &Config,
) -> Result<String, &'static str> {
    let mut unit = gateway_unit(service);
    if let Some(pool) = &config.managed {
        let source = pool
            .workers
            .first()
            .ok_or("managed_pool_configuration_invalid")?
            .models
            .to_str()
            .ok_or("managed_model_path_invalid")?;
        if !source.starts_with('/') || source.chars().any(|c| c.is_control() || c == ':') {
            return Err("managed_model_path_invalid");
        }
        let source = source
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('%', "%%");
        unit.push_str(&format!(
            "\n[Unit]\nRequiresMountsFor=\"{source}\"\n\n[Service]\nBindReadOnlyPaths=\"{source}\":/var/lib/kuno/models\n"
        ));
    }
    if let Some(image) = &config.image {
        let source = image.models.to_str().ok_or("image_model_path_invalid")?;
        if !source.starts_with('/') || source.chars().any(|c| c.is_control() || c == ':') {
            return Err("image_model_path_invalid");
        }
        let source = source
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('%', "%%");
        unit.push_str(&format!(
            "\n[Unit]\nRequiresMountsFor=\"{source}\"\n\n[Service]\nBindReadOnlyPaths=\"{source}\":/var/lib/kuno/image-models\nTimeoutStopSec=150\n"
        ));
    }
    Ok(unit)
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
    choose_private_port(original, control, |port| {
        TcpListener::bind((Ipv4Addr::LOCALHOST, port)).is_ok()
    })
}

fn choose_private_port(
    original: u16,
    control: u16,
    mut available: impl FnMut(u16) -> bool,
) -> Result<u16, &'static str> {
    // Prefer the selected ordinary layout for new installations. Existing
    // installations continue using their recorded upstream on upgrade.
    for port in
        std::iter::once(11433).chain(original.saturating_add(1)..=original.saturating_add(100))
    {
        if port == original || port == control {
            continue;
        }
        if available(port) {
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
    fn new_private_port_prefers_11433_and_skips_conflicts() {
        assert_eq!(choose_private_port(11434, 11432, |_| true), Ok(11433));
        assert_eq!(
            choose_private_port(11434, 11432, |port| port != 11433),
            Ok(11435)
        );
        assert_eq!(choose_private_port(11433, 11432, |_| true), Ok(11434));
        assert_eq!(choose_private_port(11434, 11433, |_| true), Ok(11435));
        assert_eq!(
            choose_private_port(11434, 11432, |_| false),
            Err("no_private_port_available")
        );
    }

    #[test]
    fn image_model_mount_is_read_only_and_escaped() {
        let mut config = Config {
            image: Some(crate::image_runtime::ImageConfig {
                models: "/media/private/image models %n".into(),
                pci_address: "0000:2d:00.0".into(),
                timeout_seconds: 3600,
                engine: Default::default(),
                runtime: None,
                collections: vec!["qwen-image-2512-fp8".into()],
            }),
            ..Config::default()
        };
        let unit = configured_gateway_unit("ollama.service", &config).unwrap();
        assert!(unit.contains(
            "BindReadOnlyPaths=\"/media/private/image models %%n\":/var/lib/kuno/image-models"
        ));
        for bad in ["/tmp/x:y", "/tmp/x\nExecStart=bad"] {
            config.image.as_mut().unwrap().models = bad.into();
            assert!(configured_gateway_unit("ollama.service", &config).is_err());
        }
    }

    #[test]
    fn managed_model_mount_is_scoped_and_cannot_inject_unit_directives() {
        use crate::{
            pool::PoolConfig,
            worker::{DeviceSelection, WorkerConfig},
        };
        let mut config = Config {
            managed: Some(PoolConfig {
                reserve_bytes: 1024u64.pow(3),
                workers: vec![WorkerConfig {
                    executable: "/usr/local/bin/ollama".into(),
                    models: "/media/private/models %n".into(),
                    home: "/var/lib/kuno/workers/ollama-0".into(),
                    address: "127.0.0.1:11436".parse().unwrap(),
                    device: DeviceSelection::Vulkan {
                        index: 1,
                        pci_address: "0000:2d:00.0".into(),
                    },
                    max_loaded_models: 8,
                    context_length: 4096,
                }],
            }),
            ..Config::default()
        };
        let unit = configured_gateway_unit("ollama.service", &config).unwrap();
        assert!(
            unit.contains("BindReadOnlyPaths=\"/media/private/models %%n\":/var/lib/kuno/models")
        );
        assert!(!unit.contains("SupplementaryGroups=jmicko"));
        for path in ["/media/models:other", "/media/models\nExecStart=/bad"] {
            config.managed.as_mut().unwrap().workers[0].models = path.into();
            assert_eq!(
                configured_gateway_unit("ollama.service", &config),
                Err("managed_model_path_invalid")
            );
        }
        config.managed = None;
        assert!(
            !configured_gateway_unit("ollama.service", &config)
                .unwrap()
                .contains("BindReadOnlyPaths")
        );
    }

    #[test]
    #[ignore = "requires a running user systemd manager; parses a temporary unit without starting it"]
    fn systemd_parses_model_mount_source_and_destination_separately() {
        use std::io::Write;
        let runtime = std::env::var_os("XDG_RUNTIME_DIR").expect("user runtime directory");
        let directory = std::path::PathBuf::from(runtime).join("systemd/user");
        std::fs::create_dir_all(&directory).unwrap();
        let mut unit = tempfile::Builder::new()
            .prefix("kuno-mount-parser-")
            .suffix(".service")
            .tempfile_in(&directory)
            .unwrap();
        let config: Config = serde_json::from_value(serde_json::json!({"managed":{
            "reserve_bytes":1073741824,"workers":[{
                "executable":"/usr/local/bin/ollama","models":"/media/private/models %n",
                "home":"/var/lib/kuno/workers/ollama-0","address":"127.0.0.1:11436",
                "device":{"backend":"vulkan","index":1,"pci_address":"0000:2d:00.0"},
                "max_loaded_models":8,"context_length":4096
            }]
        }}))
        .unwrap();
        unit.write_all(
            configured_gateway_unit("ollama.service", &config)
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
        unit.flush().unwrap();
        let name = unit
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let reload = || {
            std::process::Command::new("systemctl")
                .args(["--user", "daemon-reload"])
                .output()
                .unwrap()
        };
        assert!(reload().status.success());
        let output = std::process::Command::new("systemctl")
            .args([
                "--user",
                "show",
                "--value",
                "--property=BindReadOnlyPaths",
                &name,
            ])
            .output()
            .unwrap();
        // Unlink and reload before assertions, including a failing parse check.
        drop(unit);
        assert!(reload().status.success());
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).unwrap().trim(),
            "/media/private/models %n:/var/lib/kuno/models:rbind"
        );
    }

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
