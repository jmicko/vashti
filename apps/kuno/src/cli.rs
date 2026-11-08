//! Interactive compatibility commands use the original Ollama client, with an
//! explicitly selected Kuno endpoint. The CLI never inherits OLLAMA_HOST as its
//! own destination and does not copy arguments or responses into diagnostics.
use std::{
    ffi::{OsStr, OsString},
    fs,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    os::unix::{fs::MetadataExt, fs::PermissionsExt, process::CommandExt},
    path::{Path, PathBuf},
    process::Command,
};

const INSTALLED_CONFIG: &str = "/etc/kuno/config.json";
const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:11434";

struct Invocation {
    host: Option<OsString>,
    config: Option<PathBuf>,
    arguments: Vec<OsString>,
}

impl Invocation {
    fn parse(arguments: &[OsString]) -> Result<Option<Self>, &'static str> {
        let mut host = None;
        let mut config = None;
        let mut index = 0;
        while let Some(argument) = arguments.get(index) {
            let destination = match argument.to_str() {
                Some("--host") => &mut host,
                Some("--config") => &mut config,
                _ => break,
            };
            if destination.is_some() {
                return Err("duplicate_cli_endpoint_option");
            }
            *destination = Some(
                arguments
                    .get(index + 1)
                    .ok_or("missing_cli_endpoint_option_value")?
                    .clone(),
            );
            index += 2;
        }
        let supported = arguments
            .get(index)
            .and_then(|s| s.to_str())
            .is_some_and(|command| {
                matches!(
                    command,
                    "run"
                        | "list"
                        | "ls"
                        | "ps"
                        | "show"
                        | "pull"
                        | "push"
                        | "stop"
                        | "cp"
                        | "rm"
                        | "create"
                )
            });
        if !supported {
            return if host.is_some() || config.is_some() {
                Err("cli_endpoint_options_require_inference_command")
            } else {
                Ok(None)
            };
        }
        Ok(Some(Self {
            host,
            config: config.map(PathBuf::from),
            // After the command, every argument belongs to Ollama. In particular,
            // prompt text and flags named --host/--config must remain untouched.
            arguments: arguments[index..].to_vec(),
        }))
    }
}

/// Exec before starting Tokio threads. Terminal descriptors, signal handling,
/// cancellation and exit status belong to the existing Ollama CLI. No deadline
/// is introduced. Success replaces this process; errors contain no arguments.
pub fn run_if_requested(arguments: &[OsString]) -> Result<bool, &'static str> {
    let Some(invocation) = Invocation::parse(arguments)? else {
        return Ok(false);
    };
    let endpoint = select_endpoint(
        invocation.host.as_deref(),
        std::env::var_os("KUNO_HOST").as_deref(),
        invocation.config.as_deref(),
        Path::new(INSTALLED_CONFIG),
    )?;
    let executable = find_ollama(
        std::env::var_os("PATH").as_deref(),
        &std::env::current_exe().map_err(|_| "cli_executable_unavailable")?,
    )?;
    crate::image_privacy::protect_process()?;
    let _ = Command::new(executable)
        .args(invocation.arguments)
        .env("OLLAMA_HOST", endpoint)
        // Official Ollama v0.35.1 cmd/interactive.go uses this variable to
        // disable readline history. Never inherit a user's contrary default.
        .env("OLLAMA_NOHISTORY", "1")
        .exec();
    Err("ollama_cli_execution_failed")
}

fn select_endpoint(
    explicit: Option<&OsStr>,
    environment: Option<&OsStr>,
    config: Option<&Path>,
    installed: &Path,
) -> Result<String, &'static str> {
    if let Some(host) = explicit.or(environment) {
        return normalize_endpoint(host.to_str().ok_or("cli_endpoint_invalid")?);
    }
    let selected = config.unwrap_or(installed);
    let bytes = match fs::read(selected) {
        Ok(bytes) if bytes.len() <= 1024 * 1024 => bytes,
        Ok(_) => return Err("cli_configuration_invalid"),
        Err(error) if config.is_none() && error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(DEFAULT_ENDPOINT.to_owned());
        }
        Err(_) => return Err("cli_configuration_read_failed"),
    };
    // Read only the public inference listener. Unrelated worker, credential and
    // privileged control fields are not part of the client's endpoint choice.
    #[derive(serde::Deserialize)]
    struct Listener {
        listen: SocketAddr,
    }
    let listener: Listener =
        serde_json::from_slice(&bytes).map_err(|_| "cli_configuration_invalid")?;
    let address = match listener.listen.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), listener.listen.port())
        }
        IpAddr::V6(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv6Addr::LOCALHOST.into(), listener.listen.port())
        }
        _ => listener.listen,
    };
    normalize_endpoint(&format!("http://{address}"))
}

fn normalize_endpoint(input: &str) -> Result<String, &'static str> {
    if input.is_empty()
        || input.chars().any(|c| c.is_whitespace() || c.is_control())
        || input.contains(['\\', '@', '?', '#'])
    {
        return Err("cli_endpoint_invalid");
    }
    let candidate = if input.contains("://") {
        input.to_owned()
    } else {
        format!("http://{input}")
    };
    let (_, authority) = candidate.split_once("://").ok_or("cli_endpoint_invalid")?;
    if authority.is_empty()
        || authority
            .find('/')
            .is_some_and(|offset| &authority[offset..] != "/")
    {
        return Err("cli_endpoint_invalid");
    }
    let mut url = reqwest::Url::parse(&candidate).map_err(|_| "cli_endpoint_invalid")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port() == Some(0)
        || url.path() != "/"
    {
        return Err("cli_endpoint_invalid");
    }
    let replacement = match url.host_str() {
        Some("0.0.0.0") => Some("127.0.0.1"),
        Some("[::]") => Some("[::1]"),
        _ => None,
    };
    if let Some(host) = replacement {
        url.set_host(Some(host))
            .map_err(|_| "cli_endpoint_invalid")?;
    }
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

fn find_ollama(path: Option<&OsStr>, own_executable: &Path) -> Result<PathBuf, &'static str> {
    let own = fs::metadata(own_executable).map_err(|_| "cli_executable_unavailable")?;
    let mut recursive = false;
    for directory in path.into_iter().flat_map(std::env::split_paths) {
        let executable = directory.join("ollama");
        let Ok(metadata) = fs::metadata(&executable) else {
            continue;
        };
        if !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
            continue;
        }
        if (metadata.dev(), metadata.ino()) == (own.dev(), own.ino()) {
            recursive = true;
            continue;
        }
        return Ok(executable);
    }
    Err(if recursive {
        "ollama_cli_cannot_be_kuno"
    } else {
        "ollama_cli_not_found_install_ollama_or_add_it_to_path"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn only_prefix_options_are_consumed_and_administration_stays_in_kuno() {
        let values = args(&[
            "--host",
            "host:1234",
            "run",
            "fixture",
            "--config",
            "prompt",
        ]);
        let invocation = Invocation::parse(&values).unwrap().unwrap();
        assert_eq!(
            invocation.arguments,
            args(&["run", "fixture", "--config", "prompt"])
        );
        assert_eq!(invocation.host.as_deref(), Some(OsStr::new("host:1234")));
        for command in ["serve", "install", "help", "image-worker", "resources"] {
            assert!(Invocation::parse(&args(&[command])).unwrap().is_none());
        }
        assert!(Invocation::parse(&args(&["--host", "host:1234", "serve"])).is_err());
        assert!(Invocation::parse(&args(&["--host"])).is_err());
        assert!(Invocation::parse(&args(&["--host", "a", "--host", "b", "ps"])).is_err());
    }

    #[test]
    fn explicit_environment_selected_config_installed_then_default_precedence() {
        let directory = tempfile::tempdir().unwrap();
        let installed = directory.path().join("installed.json");
        let selected = directory.path().join("selected.json");
        fs::write(&installed, r#"{"listen":"0.0.0.0:12000","other":true}"#).unwrap();
        fs::write(&selected, r#"{"listen":"[::]:12001"}"#).unwrap();
        let choose = |host, environment, config| {
            select_endpoint(host, environment, config, &installed).unwrap()
        };
        assert_eq!(
            choose(
                Some(OsStr::new("host:12002")),
                Some(OsStr::new("ignored:9")),
                Some(&selected)
            ),
            "http://host:12002"
        );
        assert_eq!(
            choose(None, Some(OsStr::new("host:12003")), Some(&selected)),
            "http://host:12003"
        );
        assert_eq!(choose(None, None, Some(&selected)), "http://[::1]:12001");
        assert_eq!(choose(None, None, None), "http://127.0.0.1:12000");
        fs::remove_file(&installed).unwrap();
        assert_eq!(choose(None, None, None), DEFAULT_ENDPOINT);
        assert!(select_endpoint(None, None, Some(&installed), &installed).is_err());
        fs::write(&installed, "{}").unwrap();
        assert!(select_endpoint(None, None, None, &installed).is_err());
    }

    #[test]
    fn endpoint_validation_rejects_credentials_paths_and_url_ambiguities() {
        for input in [
            "",
            " host:1234",
            "http://",
            "http:///host:1234",
            "ftp://host:1234",
            "http://user:secret@host:1234",
            "http://@host:1234",
            "http://host:0",
            "http://host:1234/a",
            "http://host:1234/a/..",
            "http://host:1234/?key=secret",
            "http://host:1234/#secret",
            "http://host\\other:1234",
            "host:1234\n",
        ] {
            assert_eq!(normalize_endpoint(input), Err("cli_endpoint_invalid"));
        }
        for (input, expected) in [
            ("0.0.0.0:11434", "http://127.0.0.1:11434"),
            ("http://[::]:11434/", "http://[::1]:11434"),
            (
                "https://inference.example:8443/",
                "https://inference.example:8443",
            ),
        ] {
            assert_eq!(normalize_endpoint(input).unwrap(), expected);
        }
    }

    #[test]
    fn executable_lookup_skips_direct_symlink_and_hardlink_recursion() {
        let directory = tempfile::tempdir().unwrap();
        let own = directory.path().join("kuno");
        fs::write(&own, []).unwrap();
        fs::set_permissions(&own, fs::Permissions::from_mode(0o755)).unwrap();
        let executable = directory.path().join("ollama");
        std::os::unix::fs::symlink(&own, &executable).unwrap();
        assert_eq!(
            find_ollama(Some(directory.path().as_os_str()), &own),
            Err("ollama_cli_cannot_be_kuno")
        );
        fs::remove_file(&executable).unwrap();
        fs::hard_link(&own, &executable).unwrap();
        assert_eq!(
            find_ollama(Some(directory.path().as_os_str()), &own),
            Err("ollama_cli_cannot_be_kuno")
        );
        fs::remove_file(&executable).unwrap();
        fs::write(&executable, []).unwrap();
        assert_eq!(
            find_ollama(Some(directory.path().as_os_str()), &own),
            Err("ollama_cli_not_found_install_ollama_or_add_it_to_path")
        );
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(
            find_ollama(Some(directory.path().as_os_str()), &own).unwrap(),
            executable
        );
    }
}
