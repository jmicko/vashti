//! Linux service adoption and privileged, installation-scoped recovery.
//! The normal gateway remains unprivileged.
use crate::{
    config::Config,
    deployment::{Artifact, Host, Phase, Transaction},
    storage, takeover,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, &'static str>;
const SETUP: &str = "/var/lib/kuno-setup";
pub(crate) const READY: &str = "/etc/kuno/activation-ready";
const BINARY: &str = "/usr/local/lib/kuno/kuno";
const GATEWAY: &str = "kuno.service";
const RECOVERY_TIMER: &str = "kuno-recovery.timer";
const RECOVERY_PENDING: &str = "/var/lib/kuno-setup/auto-recovery-pending";
const RECOVERY_SERVICE_UNIT: &str = "[Unit]\nDescription=Recover Kuno owned inference services\n[Service]\nType=oneshot\nExecStart=/usr/local/lib/kuno/kuno auto-recover\nTimeoutStartSec=360\nUMask=0077\nLimitCORE=0\n";
const RECOVERY_TIMER_UNIT: &str = "[Unit]\nDescription=Check Kuno inference recovery\n[Timer]\nOnBootSec=30\nOnUnitInactiveSec=30\nAccuracySec=1\n[Install]\nWantedBy=timers.target\n";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Settings {
    host: Option<SocketAddr>,
    max_loaded: Option<u32>,
    request_logging: bool,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Installation {
    version: u32,
    service: String,
    original_listen: SocketAddr,
    original_executable: PathBuf,
    original_settings: Settings,
    ollama_version: String,
    config: Config,
    token: String,
}

#[derive(Clone, Debug)]
struct Socket {
    address: SocketAddr,
    state: u8,
    inode: u64,
}

/// No command output is logged. Bound both execution and captured bytes; never
/// spill process environment or service diagnostics to temporary files.
pub(crate) fn command(args: &[&str]) -> Result<String> {
    let mut child = Command::new("/usr/bin/systemctl")
        .args(args)
        .env("SYSTEMD_PAGER", "")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| "systemctl_unavailable")?;
    let stdout = child.stdout.take().ok_or("systemctl_capture_failed")?;
    let reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|_| "systemctl_wait_failed")? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err("systemctl_timeout");
        }
        thread::sleep(Duration::from_millis(25));
    };
    let bytes = reader
        .join()
        .map_err(|_| "systemctl_capture_failed")?
        .map_err(|_| "systemctl_capture_failed")?;
    if !status.success() {
        return Err("systemctl_failed");
    }
    if bytes.len() > 1024 * 1024 {
        return Err("systemctl_output_too_large");
    }
    String::from_utf8(bytes).map_err(|_| "systemctl_encoding_invalid")
}

fn sockets() -> Result<Vec<Socket>> {
    let mut sockets = Vec::new();
    for (path, v6) in [("/proc/net/tcp", false), ("/proc/net/tcp6", true)] {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if v6 && e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => return Err("socket_inventory_unavailable"),
        };
        for line in text.lines().skip(1) {
            sockets.push(parse_socket(line, v6)?);
        }
    }
    Ok(sockets)
}

fn parse_socket(line: &str, v6: bool) -> Result<Socket> {
    let columns: Vec<_> = line.split_ascii_whitespace().collect();
    if columns.len() < 10 {
        return Err("socket_inventory_invalid");
    }
    let (address, port) = columns[1]
        .split_once(':')
        .ok_or("socket_inventory_invalid")?;
    let mut bytes = Vec::new();
    if address.len() != if v6 { 32 } else { 8 } {
        return Err("socket_inventory_invalid");
    }
    for word in address.as_bytes().chunks(8) {
        let word = std::str::from_utf8(word).map_err(|_| "socket_inventory_invalid")?;
        bytes.extend(
            u32::from_str_radix(word, 16)
                .map_err(|_| "socket_inventory_invalid")?
                .to_ne_bytes(),
        );
    }
    let ip = if v6 {
        IpAddr::V6(Ipv6Addr::from(
            <[u8; 16]>::try_from(bytes).map_err(|_| "socket_inventory_invalid")?,
        ))
    } else {
        IpAddr::V4(Ipv4Addr::from(
            <[u8; 4]>::try_from(bytes).map_err(|_| "socket_inventory_invalid")?,
        ))
    };
    Ok(Socket {
        address: SocketAddr::new(
            ip,
            u16::from_str_radix(port, 16).map_err(|_| "socket_inventory_invalid")?,
        ),
        state: u8::from_str_radix(columns[3], 16).map_err(|_| "socket_inventory_invalid")?,
        inode: columns[9].parse().map_err(|_| "socket_inventory_invalid")?,
    })
}

pub(crate) fn observed_listener(configured: SocketAddr) -> Result<SocketAddr> {
    let listeners: Vec<_> = sockets()?
        .into_iter()
        .filter(|s| s.state == 10 && s.address.port() == configured.port())
        .collect();
    if listeners.len() != 1 {
        return Err("ollama_listener_missing_or_ambiguous");
    }
    let actual = listeners[0].address;
    if actual == configured {
        return Ok(actual);
    }
    if configured.ip().is_unspecified()
        && actual.ip().is_unspecified()
        && actual.is_ipv6()
        && fs::read_to_string("/proc/sys/net/ipv6/bindv6only")
            .map_err(|_| "dual_stack_setting_unavailable")?
            .trim()
            == "0"
    {
        return Ok(actual);
    }
    Err("configured_listener_does_not_match_actual_listener")
}

fn pid(service: &str) -> Result<u32> {
    let pid = takeover::property(service, "MainPID")?
        .trim()
        .parse()
        .map_err(|_| "service_pid_invalid")?;
    if pid == 0 {
        return Err("service_not_running");
    }
    Ok(pid)
}

pub(crate) fn owns_listener(pid: u32, address: SocketAddr) -> Result<()> {
    let mut owned = BTreeSet::new();
    for entry in
        fs::read_dir(format!("/proc/{pid}/fd")).map_err(|_| "listener_ownership_requires_root")?
    {
        let entry = entry.map_err(|_| "process_descriptor_unreadable")?;
        if let Ok(target) = fs::read_link(entry.path())
            && let Some(inode) = target
                .to_str()
                .and_then(|s| s.strip_prefix("socket:["))
                .and_then(|s| s.strip_suffix(']'))
                .and_then(|s| s.parse::<u64>().ok())
        {
            owned.insert(inode);
        }
    }
    let listeners: Vec<_> = sockets()?
        .into_iter()
        .filter(|s| s.state == 10 && s.address.port() == address.port())
        .collect();
    if listeners.len() != 1
        || listeners[0].address != address
        || !owned.contains(&listeners[0].inode)
    {
        return Err("listener_not_owned_by_expected_service");
    }
    Ok(())
}

fn settings(pid: u32) -> Result<Settings> {
    let bytes = fs::read(format!("/proc/{pid}/environ"))
        .map_err(|_| "process_environment_requires_root")?;
    let mut settings = Settings {
        host: None,
        max_loaded: None,
        request_logging: false,
    };
    for entry in bytes.split(|b| *b == 0) {
        if let Some(value) = entry.strip_prefix(b"OLLAMA_HOST=") {
            settings.host = Some(takeover::parse_listen(
                std::str::from_utf8(value).map_err(|_| "process_host_invalid")?,
            )?);
        } else if let Some(value) = entry.strip_prefix(b"OLLAMA_MAX_LOADED_MODELS=") {
            settings.max_loaded = Some(
                std::str::from_utf8(value)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .ok_or("process_model_limit_invalid")?,
            );
        } else if let Some(value) = entry.strip_prefix(b"OLLAMA_DEBUG_LOG_REQUESTS=") {
            settings.request_logging = match value {
                b"1" | b"true" => true,
                b"0" | b"false" | b"" => false,
                _ => return Err("request_logging_setting_invalid"),
            };
        }
    }
    Ok(settings)
}

fn require_root() -> Result<()> {
    // SAFETY: geteuid has no pointer arguments or process side effects.
    if unsafe { libc::geteuid() } != 0 {
        return Err("root_required_run_this_command_with_sudo");
    }
    Ok(())
}

fn loopback(address: SocketAddr) -> SocketAddr {
    if address.ip().is_unspecified() {
        SocketAddr::new(
            if address.is_ipv6() {
                IpAddr::V6(Ipv6Addr::LOCALHOST)
            } else {
                IpAddr::V4(Ipv4Addr::LOCALHOST)
            },
            address.port(),
        )
    } else {
        address
    }
}

fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .pool_max_idle_per_host(0)
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|_| "health_client_failed")
}

fn json(
    address: SocketAddr,
    path: &str,
    token: Option<&str>,
    post: bool,
) -> Result<serde_json::Value> {
    let client = client()?;
    let url = format!("http://{}{path}", loopback(address));
    let mut request = if post {
        client.post(url)
    } else {
        client.get(url)
    };
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let response = request.send().map_err(|_| "health_request_failed")?;
    if !response.status().is_success() {
        return Err("health_response_unsuccessful");
    }
    let mut bytes = Vec::new();
    response
        .take(65537)
        .read_to_end(&mut bytes)
        .map_err(|_| "health_body_failed")?;
    if bytes.len() > 65536 {
        return Err("health_body_too_large");
    }
    serde_json::from_slice(&bytes).map_err(|_| "health_json_invalid")
}

fn version(address: SocketAddr) -> Result<String> {
    let value = json(address, "/api/version", None, false)?;
    value
        .get("version")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty() && s.len() <= 128)
        .map(str::to_owned)
        .ok_or("ollama_version_invalid")
}

fn wait(mut check: impl FnMut() -> Result<()>) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(45);
    loop {
        match check() {
            Ok(()) => return Ok(()),
            Err(e) if Instant::now() >= deadline => return Err(e),
            Err(_) => {}
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn action(service: &str, verb: &str, active: bool) -> Result<()> {
    command(&[verb, "--no-block", service])?;
    wait(|| {
        let state = takeover::property(service, "ActiveState")?;
        if (active && state.trim() == "active")
            || (!active && matches!(state.trim(), "inactive" | "failed"))
        {
            Ok(())
        } else {
            Err("service_transition_timeout")
        }
    })
}

fn trusted_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::metadata(path).map_err(|_| "service_file_unavailable")?;
    if !metadata.is_file() || metadata.uid() != 0 || metadata.mode() & 0o022 != 0 {
        return Err("service_file_not_root_controlled");
    }
    Ok(())
}

fn capture(service: &str, config: &Config) -> Result<Installation> {
    let plan = takeover::inspect(service, config)?;
    if service == GATEWAY || plan.active_state != "active" {
        return Err("installation_requires_running_ollama_service");
    }
    let process = pid(service)?;
    owns_listener(process, plan.original_listen)?;
    let executable = fs::read_link(format!("/proc/{process}/exe"))
        .map_err(|_| "service_executable_unreadable")?;
    trusted_file(&executable)?;
    if executable.file_name().and_then(|n| n.to_str()) != Some("ollama") {
        return Err("service_executable_is_not_ollama");
    }
    let arguments =
        fs::read(format!("/proc/{process}/cmdline")).map_err(|_| "service_arguments_unreadable")?;
    let args: Vec<_> = arguments
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .collect();
    if args.len() != 2 || args[1] != b"serve" {
        return Err("unsupported_ollama_service_command");
    }
    let fragment = takeover::property(service, "FragmentPath")?;
    trusted_file(Path::new(fragment.trim()))?;
    for path in takeover::property(service, "DropInPaths")?.split_ascii_whitespace() {
        trusted_file(Path::new(path))?;
    }
    if takeover::property(service, "KillMode")?.trim() != "control-group" {
        return Err("ollama_service_requires_control_group_kill_mode");
    }
    if fs::read_link(format!("/proc/{process}/ns/net")).ok()
        != fs::read_link("/proc/self/ns/net").ok()
    {
        return Err("ollama_network_namespace_unsupported");
    }
    let original_settings = settings(process)?;
    let env = takeover::parse_environment(&takeover::property(service, "Environment")?)?;
    if original_settings.max_loaded
        != env
            .get("OLLAMA_MAX_LOADED_MODELS")
            .map(|s| s.parse::<u32>())
            .transpose()
            .map_err(|_| "model_limit_invalid")?
    {
        return Err("ollama_effective_environment_differs_from_unit");
    }
    let mut random = [0u8; 32];
    fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut random))
        .map_err(|_| "credential_generation_failed")?;
    let mut config = plan.proposed_config;
    config.state_dir = "/var/lib/kuno".into();
    config.activation_file = Some(READY.into());
    Ok(Installation {
        version: 1,
        service: service.into(),
        original_listen: plan.original_listen,
        original_executable: executable,
        original_settings,
        ollama_version: version(plan.original_listen)?,
        config,
        token: random.iter().map(|b| format!("{b:02x}")).collect(),
    })
}

pub(crate) fn ollama_drop_in(upstream: SocketAddr) -> String {
    format!(
        "[Service]\nEnvironment=\"OLLAMA_HOST={}\"\nEnvironment=\"OLLAMA_DEBUG_LOG_REQUESTS=false\"\nStandardOutput=null\nStandardError=null\nLimitCORE=0\nMemorySwapMax=0\n",
        upstream
    )
}

fn artifacts(installation: &Installation, executable: Vec<u8>) -> Result<Vec<Artifact>> {
    let drop_in = ollama_drop_in(installation.config.upstream);
    Ok(vec![
        Artifact {
            path: "etc/systemd/system/kuno-recovery.service".into(),
            bytes: RECOVERY_SERVICE_UNIT.as_bytes().to_vec(),
            mode: 0o644,
        },
        Artifact {
            path: "etc/systemd/system/kuno-recovery.timer".into(),
            bytes: RECOVERY_TIMER_UNIT.as_bytes().to_vec(),
            mode: 0o644,
        },
        Artifact {
            path: "etc/kuno/config.json".into(),
            bytes: serde_json::to_vec_pretty(&installation.config)
                .map_err(|_| "config_encoding_failed")?,
            mode: 0o644,
        },
        Artifact {
            path: "etc/kuno/control-token".into(),
            bytes: installation.token.as_bytes().to_vec(),
            mode: 0o600,
        },
        Artifact {
            path: "etc/systemd/system/kuno.service".into(),
            bytes: takeover::configured_gateway_unit(&installation.service, &installation.config)?
                .into_bytes(),
            mode: 0o644,
        },
        Artifact {
            path: format!(
                "etc/systemd/system/{}.d/zzzz-kuno.conf",
                installation.service
            )
            .into(),
            bytes: drop_in.into_bytes(),
            mode: 0o644,
        },
        Artifact {
            path: BINARY.trim_start_matches('/').into(),
            bytes: executable,
            mode: 0o755,
        },
    ])
}

struct SystemdHost {
    installation: Installation,
}

impl SystemdHost {
    fn check_original(&self) -> Result<()> {
        let i = &self.installation;
        let process = pid(&i.service)?;
        if fs::read_link(format!("/proc/{process}/exe"))
            .map_err(|_| "service_executable_unreadable")?
            != i.original_executable
        {
            return Err("ollama_executable_changed");
        }
        owns_listener(process, i.original_listen)?;
        if settings(process)? != i.original_settings {
            return Err("ollama_environment_changed");
        }
        Ok(())
    }
    fn status(&self) -> Result<serde_json::Value> {
        json(
            self.installation.config.control_listen,
            "/kuno/v1/status",
            Some(&self.installation.token),
            false,
        )
    }
}

impl Host for SystemdHost {
    fn preflight(&mut self) -> Result<()> {
        require_root()?;
        self.check_original()?;
        if takeover::property(GATEWAY, "LoadState")?.trim() != "not-found" {
            return Err("kuno_service_already_exists");
        }
        use std::os::unix::fs::PermissionsExt;
        for path in ["/etc/kuno", "/usr/local/lib/kuno"] {
            if let Ok(metadata) = fs::metadata(path)
                && metadata.permissions().mode() & 0o005 != 0o005
            {
                return Err("existing_installation_directory_not_accessible_to_service");
            }
        }
        for entry in
            fs::read_dir("/etc/systemd/system").map_err(|_| "systemd_directory_unreadable")?
        {
            let entry = entry.map_err(|_| "systemd_directory_unreadable")?;
            if entry
                .file_type()
                .map_err(|_| "systemd_directory_unreadable")?
                .is_dir()
                && fs::symlink_metadata(entry.path().join(GATEWAY)).is_ok()
            {
                return Err("preexisting_kuno_enable_link_requires_manual_review");
            }
        }
        for address in [
            self.installation.config.upstream,
            self.installation.config.control_listen,
        ] {
            TcpListener::bind(address).map_err(|_| "installation_port_occupied")?;
        }
        if fs::symlink_metadata(READY).is_ok() {
            return Err("activation_marker_already_exists");
        }
        if Path::new("/var/lib/private/kuno/upstream-in-flight").exists()
            || Path::new("/var/lib/kuno/upstream-in-flight").exists()
        {
            return Err("upstream_recovery_required_before_installation");
        }
        Ok(())
    }
    fn original_active(&mut self) -> Result<bool> {
        Ok(takeover::property(&self.installation.service, "ActiveState")?.trim() == "active")
    }
    fn stop_original(&mut self) -> Result<()> {
        eprintln!("kuno: Stopping Ollama and its model runners.");
        action(&self.installation.service, "stop", false)?;
        clear_stopped_upstream_marker()
    }
    fn reload(&mut self) -> Result<()> {
        command(&["daemon-reload"]).map(|_| ())
    }
    fn start_private_upstream(&mut self) -> Result<()> {
        action(&self.installation.service, "start", true)
    }
    fn verify_private_upstream(&mut self) -> Result<()> {
        eprintln!("kuno: Checking the private Ollama listener (up to 45 seconds).");
        wait(|| {
            let i = &self.installation;
            let process = pid(&i.service)?;
            owns_listener(process, i.config.upstream)?;
            let actual = settings(process)?;
            if actual.host != Some(i.config.upstream)
                || actual.max_loaded != i.original_settings.max_loaded
                || actual.request_logging
            {
                return Err("private_upstream_environment_incorrect");
            }
            if version(i.config.upstream)? != i.ollama_version {
                return Err("ollama_version_changed");
            }
            Ok(())
        })
    }
    fn start_gateway(&mut self) -> Result<()> {
        action(GATEWAY, "start", true)
    }
    fn verify_gateway(&mut self) -> Result<()> {
        eprintln!("kuno: Checking Kuno startup and configured GPU workers (up to 45 seconds).");
        wait(|| {
            let i = &self.installation;
            let process = pid(GATEWAY)?;
            owns_listener(process, i.config.listen)?;
            owns_listener(process, i.config.control_listen)?;
            let status = self.status()?;
            let runtime_config = installed_config(&i.config)?;
            if status["mode"]
                != (if runtime_config.managed.is_some() {
                    "managed_experimental"
                } else {
                    "gateway_only"
                })
                || status["admission"]["recovery_required"] != false
            {
                return Err("gateway_not_ready");
            }
            if version(i.config.listen)? != i.ollama_version {
                return Err("gateway_upstream_mismatch");
            }
            if i.config.listen
                == SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), i.config.listen.port())
                && version(SocketAddr::new(
                    IpAddr::V4(Ipv4Addr::LOCALHOST),
                    i.config.listen.port(),
                ))? != i.ollama_version
            {
                return Err("gateway_ipv4_compatibility_failed");
            }
            Ok(())
        })
    }
    fn stop_gateway(&mut self) -> Result<()> {
        let load = takeover::property(GATEWAY, "LoadState")?;
        if load.trim() == "not-found" {
            remove_activation()?;
            return Ok(());
        }
        if takeover::property(GATEWAY, "FragmentPath")?.trim() != "/etc/systemd/system/kuno.service"
            || !takeover::property(GATEWAY, "DropInPaths")?
                .trim()
                .is_empty()
        {
            return Err("gateway_service_changed_restore_refused");
        }
        // Hold the setup transaction lock while disabling the supervisor. A
        // concurrently scheduled check cannot enter the recovery mutation path.
        if Path::new("/etc/systemd/system/kuno-recovery.timer").exists() {
            command(&["disable", "--now", RECOVERY_TIMER])?;
        }
        clear_recovery_pending()?;
        eprintln!("kuno: Stopping Kuno and its GPU workers.");
        remove_activation()?;
        // systemd stops the listener even if control is unavailable. SIGTERM
        // pauses admission and allows a bounded drain before forced shutdown.
        // The transaction then stops Ollama's runner group before clearing any
        // uncertain in-flight marker or reopening the original listener.
        // An interrupted restore may already have removed the unit after
        // disabling it. Cached systemd state can still report it as loaded.
        if Path::new("/etc/systemd/system/kuno.service").exists() {
            command(&["disable", GATEWAY])?;
        }
        action(GATEWAY, "stop", false)
    }
    fn restore_original(&mut self, was_active: bool) -> Result<()> {
        eprintln!("kuno: Restoring the original Ollama service state.");
        if was_active {
            action(&self.installation.service, "start", true)
        } else {
            action(&self.installation.service, "stop", false)
        }
    }
    fn verify_original(&mut self, was_active: bool) -> Result<()> {
        if !was_active {
            return Ok(());
        }
        wait(|| {
            self.check_original()?;
            version(self.installation.original_listen).map(|_| ())
        })
    }
    fn activate_gateway(&mut self) -> Result<()> {
        command(&["enable", GATEWAY])?;
        if Path::new("/etc/systemd/system/kuno-recovery.timer").exists() {
            command(&["enable", "--now", RECOVERY_TIMER])?;
        }
        if activation_ready(Path::new(READY))? {
            return Ok(());
        }
        json(
            self.installation.config.control_listen,
            "/kuno/v1/admission/resume",
            Some(&self.installation.token),
            true,
        )?;
        if self.status()?["admission"]["paused"] != false {
            return Err("gateway_activation_failed");
        }
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(READY)
            .map_err(|_| "activation_marker_write_failed")?;
        file.sync_all()
            .map_err(|_| "activation_marker_write_failed")?;
        fs::File::open("/etc/kuno")
            .and_then(|f| f.sync_all())
            .map_err(|_| "activation_marker_write_failed")
    }
}

fn remove_activation() -> Result<()> {
    match fs::symlink_metadata(READY) {
        Ok(m) if !m.is_file() || m.len() != 0 => Err("activation_marker_changed"),
        Ok(_) => fs::remove_file(READY).map_err(|_| "activation_marker_remove_failed"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("activation_marker_unreadable"),
    }
}

pub fn activation_ready(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(m) if m.is_file() && m.len() == 0 => Ok(true),
        Ok(_) => Err("activation_marker_changed"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(_) => Err("activation_marker_unreadable"),
    }
}

fn load_installation() -> Result<Installation> {
    let path = Path::new(SETUP).join("installation.json");
    trusted_file(&path)?;
    let bytes = fs::read(path).map_err(|_| "installation_record_unreadable")?;
    let installation: Installation =
        serde_json::from_slice(&bytes).map_err(|_| "installation_record_invalid")?;
    installation.config.validate()?;
    if installation.version != 1
        || installation.service == GATEWAY
        || !installation.service.ends_with(".service")
        || !installation
            .service
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-@".contains(&b))
        || installation.service.starts_with('-')
        || installation.token.len() != 64
        || !installation.token.bytes().all(|b| b.is_ascii_hexdigit())
        || installation.config.activation_file.as_deref() != Some(Path::new(READY))
    {
        return Err("installation_record_invalid");
    }
    Ok(installation)
}

// The setup record preserves original service identity. The owned config is
// the transactional source of current managed mode, including after rollback.
fn installed_config(original: &Config) -> Result<Config> {
    let current = Config::read(Path::new("/etc/kuno/config.json"))?;
    let mut expected = original.clone();
    expected.managed = None;
    expected.image = None;
    let mut actual = current.clone();
    actual.managed = None;
    actual.image = None;
    if serde_json::to_value(expected).map_err(|_| "config_encoding_failed")?
        != serde_json::to_value(actual).map_err(|_| "config_encoding_failed")?
    {
        return Err("installation_identity_changed");
    }
    Ok(current)
}

fn current_installation() -> Result<Installation> {
    let mut installation = load_installation()?;
    installation.config = installed_config(&installation.config)?;
    Ok(installation)
}

/// Configuration is deployed with the binary/unit as one reversible file set.
/// No configuration edit bypasses ownership checks or service health checks.
pub fn configure_managed(mut pool: Option<crate::pool::PoolConfig>) -> Result<()> {
    require_root()?;
    let transaction = Transaction::open(Path::new("/"), Path::new(SETUP))?;
    transaction.verify_installed()?;
    let mut installation = current_installation()?;
    if let Some(pool) = &mut pool {
        for (index, worker) in pool.workers.iter_mut().enumerate() {
            trusted_file(&worker.executable)?;
            if !worker.models.is_dir() {
                return Err("managed_models_directory_unavailable");
            }
            worker.home = PathBuf::from(format!("/var/lib/kuno/workers/ollama-{index}"));
        }
    }
    installation.config.managed = pool;
    installation.config.validate()?;
    let executable = fs::read(std::env::current_exe().map_err(|_| "executable_unavailable")?)
        .map_err(|_| "executable_unavailable")?;
    transaction.replace(
        &artifacts(&installation, executable)?,
        &mut SystemdHost { installation },
    )
}

pub fn configure_image(image: Option<crate::image_runtime::ImageConfig>) -> Result<()> {
    require_root()?;
    let transaction = Transaction::open(Path::new("/"), Path::new(SETUP))?;
    transaction.verify_installed()?;
    let mut installation = current_installation()?;
    if let Some(image) = &image {
        if !cfg!(feature = "native-image") {
            return Err("image_engine_not_in_this_build");
        }
        image.validate()?;
        for id in &image.collections {
            crate::collections::verify(&image.models, id)?;
        }
    }
    installation.config.image = image;
    installation.config.validate()?;
    let executable = fs::read(std::env::current_exe().map_err(|_| "executable_unavailable")?)
        .map_err(|_| "executable_unavailable")?;
    transaction.replace(
        &artifacts(&installation, executable)?,
        &mut SystemdHost { installation },
    )
}

pub fn install(service: &str, config: &Config) -> Result<()> {
    require_root()?;
    let transaction = Transaction::open(Path::new("/"), Path::new(SETUP))?;
    let phase = transaction.phase()?;
    if phase.is_none()
        && (fs::symlink_metadata("/var/lib/kuno").is_ok()
            || fs::symlink_metadata("/var/lib/private/kuno").is_ok())
    {
        return Err("runtime_state_exists_without_installation_record");
    }
    let installation = match phase {
        Some(Phase::Complete) => current_installation()?,
        Some(Phase::Restored) | None => {
            let installation = capture(service, config)?;
            storage::write_json(&Path::new(SETUP).join("installation.json"), &installation)
                .map_err(|_| "installation_record_write_failed")?;
            installation
        }
        _ => return Err("interrupted_setup_run_takeover_restore"),
    };
    let executable = fs::read(std::env::current_exe().map_err(|_| "executable_unavailable")?)
        .map_err(|_| "executable_unavailable")?;
    transaction.apply(
        &artifacts(&installation, executable)?,
        &mut SystemdHost { installation },
    )
}

pub fn upgrade() -> Result<()> {
    require_root()?;
    let transaction = Transaction::open(Path::new("/"), Path::new(SETUP))?;
    transaction.verify_installed()?;
    let installation = current_installation()?;
    if installation.config.image.is_some() && !cfg!(feature = "native-image") {
        return Err("image_engine_not_in_this_build");
    }
    let executable = fs::read(std::env::current_exe().map_err(|_| "executable_unavailable")?)
        .map_err(|_| "executable_unavailable")?;
    transaction.replace(
        &artifacts(&installation, executable)?,
        &mut SystemdHost { installation },
    )
}

pub fn restore() -> Result<()> {
    require_root()?;
    let transaction = Transaction::open(Path::new("/"), Path::new(SETUP))?;
    if matches!(transaction.phase()?, None | Some(Phase::Restored)) {
        return Ok(());
    }
    transaction.restore(&mut SystemdHost {
        installation: load_installation()?,
    })
}

/// Explicit operator recovery: restart only the managed runtimes, then clear
/// their empty uncertainty marker once Ollama and its runner group have stopped.
pub fn recover() -> Result<()> {
    require_root()?;
    let transaction = Transaction::open(Path::new("/"), Path::new(SETUP))?;
    transaction.verify_recovery_files()?;
    recover_locked(&transaction)?;
    clear_recovery_pending()
}

fn recover_locked(transaction: &Transaction) -> Result<()> {
    let restoring = transaction.phase()? == Some(Phase::Restoring);
    let installation = load_installation()?;
    if takeover::property(GATEWAY, "FragmentPath")?.trim() != "/etc/systemd/system/kuno.service"
        || !takeover::property(GATEWAY, "DropInPaths")?
            .trim()
            .is_empty()
    {
        return Err("gateway_service_changed_recovery_refused");
    }
    recover_services(&mut ServiceRecovery {
        host: SystemdHost { installation },
        restoring,
    })
}

trait RecoverySteps {
    fn stop_gateway(&mut self) -> Result<()>;
    fn stop_upstream(&mut self) -> Result<()>;
    fn reconcile(&mut self) -> Result<()>;
    fn start_upstream(&mut self) -> Result<()>;
    fn start_gateway(&mut self) -> Result<()>;
    fn activate(&mut self) -> Result<()>;
}

fn recover_services(steps: &mut impl RecoverySteps) -> Result<()> {
    steps.stop_gateway()?;
    steps.stop_upstream()?;
    steps.reconcile()?;
    steps.start_upstream()?;
    steps.start_gateway()?;
    steps.activate()
}

struct ServiceRecovery {
    host: SystemdHost,
    restoring: bool,
}
impl RecoverySteps for ServiceRecovery {
    fn stop_gateway(&mut self) -> Result<()> {
        stop_and_verify_empty(GATEWAY)
    }
    fn stop_upstream(&mut self) -> Result<()> {
        stop_and_verify_empty(&self.host.installation.service)
    }
    fn reconcile(&mut self) -> Result<()> {
        clear_stopped_upstream_marker()
    }
    fn start_upstream(&mut self) -> Result<()> {
        self.host.start_private_upstream()?;
        self.host.verify_private_upstream()
    }
    fn start_gateway(&mut self) -> Result<()> {
        self.host.start_gateway()?;
        self.host.verify_gateway()
    }
    fn activate(&mut self) -> Result<()> {
        if self.restoring {
            Ok(())
        } else {
            self.host.activate_gateway()
        }
    }
}

// The timer runs independently of Kuno, so a failed restart is retried even
// while the gateway is down. The setup lock serializes installation, restore,
// manual recovery and automatic recovery. No inference request is replayed.
pub fn auto_recover() -> Result<()> {
    require_root()?;
    let transaction = Transaction::open(Path::new("/"), Path::new(SETUP))?;
    if transaction.phase()? != Some(Phase::Complete) {
        return Ok(());
    }
    let pending = activation_ready(Path::new(RECOVERY_PENDING))?;
    if !pending {
        // Respect an intentional systemctl stop or admission pause. A pending
        // attempt, however, must finish even if it previously stopped Kuno.
        if takeover::property(GATEWAY, "ActiveState")?.trim() != "active" {
            return Ok(());
        }
        let host = SystemdHost {
            installation: load_installation()?,
        };
        if !needs_auto_recovery(&host.status()?) {
            return Ok(());
        }
    }
    transaction.verify_recovery_files()?;
    if !pending {
        use std::os::unix::fs::OpenOptionsExt;
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(RECOVERY_PENDING)
            .map_err(|_| "auto_recovery_state_write_failed")?;
        file.sync_all()
            .map_err(|_| "auto_recovery_state_write_failed")?;
        fs::File::open(SETUP)
            .and_then(|f| f.sync_all())
            .map_err(|_| "auto_recovery_state_write_failed")?;
    }
    eprintln!("kuno: automatic_recovery_started");
    recover_locked(&transaction)?;
    clear_recovery_pending()?;
    eprintln!("kuno: automatic_recovery_completed");
    Ok(())
}

fn needs_auto_recovery(status: &serde_json::Value) -> bool {
    let admission = &status["admission"];
    admission["recovery_required"] == true
        && admission["active"] == false
        && admission["paused"] == false
}

fn clear_recovery_pending() -> Result<()> {
    match fs::remove_file(RECOVERY_PENDING) {
        Ok(()) => fs::File::open(SETUP)
            .and_then(|f| f.sync_all())
            .map_err(|_| "auto_recovery_state_sync_failed"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("auto_recovery_state_remove_failed"),
    }
}

fn stop_and_verify_empty(service: &str) -> Result<()> {
    let group = takeover::property(service, "ControlGroup")?;
    let group = group.trim();
    let relative = group.strip_prefix('/').unwrap_or(group);
    if !relative.is_empty()
        && Path::new(relative)
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("service_cgroup_invalid");
    }
    action(service, "stop", false)?;
    if relative.is_empty() {
        return Ok(());
    }
    let directory = Path::new("/sys/fs/cgroup").join(relative);
    let events = directory.join("cgroup.events");
    match fs::read_to_string(events) {
        Ok(events) if events.lines().any(|line| line == "populated 0") => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !directory.exists() => Ok(()),
        _ => Err("service_cgroup_not_empty"),
    }
}

// Call only after the gateway and Ollama's runner group have stopped. An
// interrupted request can no longer execute, so its durable marker is stale.
fn clear_stopped_upstream_marker() -> Result<()> {
    let state = match fs::canonicalize("/var/lib/kuno") {
        Ok(state) => state,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err("managed_state_directory_unavailable"),
    };
    if state != Path::new("/var/lib/private/kuno") && state != Path::new("/var/lib/kuno") {
        return Err("managed_state_directory_changed");
    }
    clear_marker_in(&state)?;
    let workers = state.join("gpu-reservations");
    match fs::symlink_metadata(&workers) {
        Ok(m) if m.is_dir() => clear_marker_in(&workers)?,
        Ok(_) => return Err("managed_worker_state_directory_changed"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err("managed_worker_state_directory_unavailable"),
    }
    Ok(())
}

fn clear_marker_in(directory: &Path) -> Result<()> {
    let marker = directory.join("upstream-in-flight");
    match fs::symlink_metadata(&marker) {
        Ok(m) if m.is_file() && m.len() == 0 => {
            fs::remove_file(marker).map_err(|_| "recovery_marker_remove_failed")?;
            fs::File::open(directory)
                .and_then(|f| f.sync_all())
                .map_err(|_| "recovery_marker_sync_failed")?;
        }
        Ok(_) => return Err("recovery_marker_changed"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err("recovery_marker_unreadable"),
    }
    Ok(())
}

pub fn doctor(service: &str, config: &Config) -> Result<serde_json::Value> {
    if Path::new(SETUP).join("takeover.json").exists() && require_root().is_ok() {
        let transaction = Transaction::open(Path::new("/"), Path::new(SETUP))?;
        let phase = transaction.phase()?;
        if phase == Some(Phase::Complete) {
            let mut host = SystemdHost {
                installation: load_installation()?,
            };
            host.verify_private_upstream()?;
            host.verify_gateway()?;
            return Ok(serde_json::json!({"installed":true,"phase":phase,"status":host.status()?}));
        }
        if !matches!(phase, None | Some(Phase::Restored)) {
            return Ok(
                serde_json::json!({"installed":false,"phase":phase,"restore_required":true}),
            );
        }
    }
    let plan = takeover::inspect(service, config)?;
    let privileged = require_root().is_ok();
    if privileged {
        let installation = capture(service, config)?;
        SystemdHost { installation }.preflight()?;
    }
    Ok(
        serde_json::json!({"installed":false,"service":service,"listener":plan.original_listen,"upstream_candidate":plan.proposed_config.upstream,
        "original_loaded_model_limit":plan.original_loaded_model_limit,"privileged_checks_passed":privileged,
        "interrupts_inference":true,"requires_root_for_installation":true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recovery_does_not_interrupt_active_paused_or_healthy_requests() {
        for recovery in [false, true] {
            for active in [false, true] {
                for paused in [false, true] {
                    let status = serde_json::json!({"admission": {
                        "recovery_required": recovery, "active": active, "paused": paused
                    }});
                    assert_eq!(needs_auto_recovery(&status), recovery && !active && !paused);
                }
            }
        }
        assert!(!needs_auto_recovery(&serde_json::json!({})));
    }

    #[derive(Default)]
    struct FakeRecovery {
        fail: Option<&'static str>,
        calls: Vec<&'static str>,
    }
    impl FakeRecovery {
        fn step(&mut self, name: &'static str) -> Result<()> {
            self.calls.push(name);
            if self.fail == Some(name) {
                Err("injected_failure")
            } else {
                Ok(())
            }
        }
    }
    impl RecoverySteps for FakeRecovery {
        fn stop_gateway(&mut self) -> Result<()> {
            self.step("stop_gateway")
        }
        fn stop_upstream(&mut self) -> Result<()> {
            self.step("stop_upstream")
        }
        fn reconcile(&mut self) -> Result<()> {
            self.step("reconcile")
        }
        fn start_upstream(&mut self) -> Result<()> {
            self.step("start_upstream")
        }
        fn start_gateway(&mut self) -> Result<()> {
            self.step("start_gateway")
        }
        fn activate(&mut self) -> Result<()> {
            self.step("activate")
        }
    }
    #[test]
    fn failed_stops_never_clear_markers_and_failed_restarts_can_retry() {
        let order = [
            "stop_gateway",
            "stop_upstream",
            "reconcile",
            "start_upstream",
            "start_gateway",
            "activate",
        ];
        for (index, fail) in order.iter().enumerate() {
            let mut steps = FakeRecovery {
                fail: Some(fail),
                ..Default::default()
            };
            assert_eq!(recover_services(&mut steps), Err("injected_failure"));
            assert_eq!(steps.calls, order[..=index]);
            steps.fail = None;
            steps.calls.clear();
            recover_services(&mut steps).unwrap();
            assert_eq!(steps.calls, order);
        }
    }

    #[test]
    fn recovery_preserves_modified_or_symlinked_uncertainty_markers() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("upstream-in-flight");
        fs::write(&marker, "changed").unwrap();
        assert_eq!(clear_marker_in(dir.path()), Err("recovery_marker_changed"));
        assert!(marker.exists());
        fs::remove_file(&marker).unwrap();
        let target = dir.path().join("target");
        fs::write(&target, "").unwrap();
        std::os::unix::fs::symlink(&target, &marker).unwrap();
        assert_eq!(clear_marker_in(dir.path()), Err("recovery_marker_changed"));
        fs::remove_file(&marker).unwrap();
        fs::write(&marker, "").unwrap();
        clear_marker_in(dir.path()).unwrap();
        clear_marker_in(dir.path()).unwrap();
        assert!(!marker.exists());
        assert!(target.exists());
    }

    #[test]
    fn proc_socket_addresses_preserve_ipv4_and_ipv6() {
        let ipv4 =
            "0: 0100007F:2CAA 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 12345";
        let socket = parse_socket(ipv4, false).unwrap();
        assert_eq!(socket.address, "127.0.0.1:11434".parse().unwrap());
        assert_eq!(socket.inode, 12345);
        let ipv6 = "0: 00000000000000000000000001000000:2CAA 00000000000000000000000000000000:0000 0A 0 0 0 1000 0 12346";
        assert_eq!(
            parse_socket(ipv6, true).unwrap().address,
            "[::1]:11434".parse().unwrap()
        );
        assert!(parse_socket("bad", false).is_err());
    }
}
