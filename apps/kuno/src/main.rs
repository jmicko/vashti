use kuno::{config::Config, gateway::Gateway, systemd, takeover};
use std::{path::Path, time::Duration};
use tokio::net::TcpListener;

const HELP: &str = "Kuno — local inference gateway (development)\n\n\
Usage:\n  kuno [--host URL] [--config FILE] run MODEL [PROMPT] [OLLAMA_OPTIONS]\n  kuno [--host URL] [--config FILE] COMMAND [ARGS]\n    Commands: list (ls), ps, show, pull, push, stop, cp, rm, create\n  kuno fetch-models COLLECTION --directory ABSOLUTE_DIRECTORY\n  kuno serve [--config FILE]\n  kuno resources [--config FILE]\n  kuno check [--config FILE]\n  kuno doctor [--service ollama.service]\n  kuno install [--yes] [--service ollama.service]\n  kuno upgrade [--yes]\n  kuno configure-managed (--workers FILE | --disable) [--yes]\n  kuno configure-image (--settings FILE | --disable) [--yes]\n  kuno install-image-runtime --directory ABSOLUTE_DIRECTORY --manifest-sha256 SHA256 [--yes]\n  kuno configure-timeouts [--response-headers SECONDS] [--stream-idle SECONDS] [--yes]\n  kuno recover [--yes]\n  kuno takeover-restore [--yes]\n  kuno takeover-plan [--config FILE] [--service ollama.service]\n  kuno takeover-bundle --output NEW_DIRECTORY [--config FILE] [--service ollama.service]\n\n\
Inference CLI endpoint: --host, KUNO_HOST, selected/installed config.listen, then 127.0.0.1:11434.\n\
The original Ollama client is required on PATH; its readline history starts disabled.\n\
OLLAMA_HOST never selects Kuno's CLI endpoint.\n\
serve requires KUNO_CONTROL_TOKEN or KUNO_CONTROL_TOKEN_FILE (32–256 URL-safe characters).\n\
configure-timeouts requires at least one timeout; 0 disables that inference deadline.\n\
Default gateway: 127.0.0.1:11433; control: 127.0.0.1:11432; upstream: 127.0.0.1:11434.\n\
takeover-plan is read-only. It never changes services, ports, or model limits.\n";

fn main() {
    match kuno::cli::run_if_requested(&std::env::args_os().skip(1).collect::<Vec<_>>()) {
        Ok(true) => return,
        Ok(false) => {}
        Err(code) => {
            eprintln!("kuno: {code}");
            std::process::exit(1);
        }
    }
    // Enter the private worker before creating any Tokio threads. All native
    // threads must inherit its filesystem policy from this initial thread.
    let result = if std::env::args().nth(1).as_deref() == Some("image-worker-launch") {
        run_image_launcher()
    } else if std::env::args().nth(1).as_deref() == Some("image-worker") {
        run_image_worker()
    } else if std::env::args().nth(1).as_deref() == Some("image-worker-resident") {
        run_resident_image_worker()
    } else {
        match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime.block_on(run()),
            Err(_) => Err("runtime_initialization_failed"),
        }
    };
    if let Err(code) = result {
        eprintln!("kuno: {code}");
        std::process::exit(1);
    }
}

fn run_image_launcher() -> Result<(), &'static str> {
    let mut args = std::env::args().skip(2);
    let id = args.next().ok_or("missing_image_runtime")?;
    let models = args.next().ok_or("missing_image_models")?;
    let pci = args.next().ok_or("missing_image_device")?;
    if args.next().is_some() {
        return Err("unknown_argument");
    }
    kuno::image_engine::launch(
        &id,
        &models,
        &pci,
        std::env::var_os("KUNO_IMAGE_PREVIEWS").is_some(),
        std::env::var_os("KUNO_IMAGE_RESIDENT").is_some(),
    )
}

#[cfg(not(feature = "native-image"))]
fn run_resident_image_worker() -> Result<(), &'static str> {
    Err("image_engine_not_in_this_build")
}

#[cfg(feature = "native-image")]
fn run_resident_image_worker() -> Result<(), &'static str> {
    // Protect the original thread before creating a session/vendor threads or
    // reading content. Model/session identity is supplied only by the gateway.
    kuno::image_privacy::protect_process()?;
    kuno::image_privacy::require_no_swap()?;
    kuno::image_privacy::restrict_filesystem()?;
    let mut args = std::env::args().skip(2);
    let directory = args.next().ok_or("missing_image_models")?;
    let pci = args.next().ok_or("missing_image_device")?;
    if args.next().is_some() {
        return Err("unknown_argument");
    }
    let mut session = kuno_image::native::Session::new(Path::new(&directory), &pci)?;
    let previews = std::env::var_os("KUNO_IMAGE_PREVIEWS").is_some();
    kuno::image_worker::serve(
        &mut std::io::stdin().lock(),
        &mut std::io::stdout(),
        |recipe| session.generate(recipe, previews),
    )
}

#[cfg(not(feature = "native-image"))]
fn run_image_worker() -> Result<(), &'static str> {
    Err("image_engine_not_in_this_build")
}

#[cfg(feature = "native-image")]
fn run_image_worker() -> Result<(), &'static str> {
    let mut args = std::env::args().skip(2);
    use std::io::{Read, Write};
    disable_core_dumps()?;
    kuno::image_privacy::require_no_swap()?;
    kuno::image_privacy::restrict_filesystem()?;
    let directory = args.next().ok_or("missing_image_models")?;
    let pci = args.next().ok_or("missing_image_device")?;
    if args.next().is_some() {
        return Err("unknown_argument");
    }
    let mut bytes = Vec::new();
    std::io::stdin()
        .take(kuno_protocol::MAX_IMAGE_REQUEST_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "image_request_read_failed")?;
    if bytes.len() > kuno_protocol::MAX_IMAGE_REQUEST_BYTES {
        return Err("image_request_too_large");
    }
    let recipe = serde_json::from_slice(&bytes).map_err(|_| "image_request_invalid")?;
    let previews = std::env::var_os("KUNO_IMAGE_PREVIEWS").is_some();
    let png = kuno_image::native::generate(Path::new(&directory), &pci, &recipe, previews)?;
    if previews {
        return kuno_image::preview::write_frame(2, &png).map_err(|_| "image_output_failed");
    }
    std::io::stdout()
        .write_all(&png)
        .map_err(|_| "image_output_failed")?;
    Ok(())
}

async fn run() -> Result<(), &'static str> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".into());
    if matches!(command.as_str(), "help" | "--help" | "-h") {
        print!("{HELP}");
        return Ok(());
    }
    if command == "auto-recover" {
        if args.next().is_some() {
            return Err("unknown_argument");
        }
        return tokio::task::spawn_blocking(systemd::auto_recover)
            .await
            .map_err(|_| "auto_recovery_task_failed")?;
    }
    if command == "fetch-models" {
        let id = args.next().ok_or("missing_collection_id")?;
        if args.next().as_deref() != Some("--directory") {
            return Err("missing_model_directory");
        }
        let directory = args.next().ok_or("missing_model_directory")?;
        if args.next().is_some() {
            return Err("unknown_argument");
        }
        tokio::task::spawn_blocking(move || kuno::collections::fetch(&id, Path::new(&directory)))
            .await
            .map_err(|_| "model_download_task_failed")??;
        println!("model_collection_ready");
        return Ok(());
    }
    if command == "configure-image" {
        let action = args.next().ok_or("missing_image_configuration")?;
        let image = match action.as_str() {
            "--disable" => None,
            "--settings" => {
                let path = args.next().ok_or("missing_image_configuration")?;
                let bytes = std::fs::read(path).map_err(|_| "image_configuration_read_failed")?;
                Some(serde_json::from_slice(&bytes).map_err(|_| "image_configuration_invalid")?)
            }
            _ => return Err("unknown_argument"),
        };
        let yes = match args.next().as_deref() {
            Some("--yes") => true,
            None => false,
            _ => return Err("unknown_argument"),
        };
        if args.next().is_some() {
            return Err("unknown_argument");
        }
        confirm_interruption(yes)?;
        tokio::task::spawn_blocking(move || systemd::configure_image(image))
            .await
            .map_err(|_| "image_configuration_task_failed")??;
        println!("image_configuration_active");
        return Ok(());
    }
    if command == "install-image-runtime" {
        if args.next().as_deref() != Some("--directory") {
            return Err("missing_runtime_directory");
        }
        let directory = args.next().ok_or("missing_runtime_directory")?;
        if args.next().as_deref() != Some("--manifest-sha256") {
            return Err("missing_runtime_manifest_checksum");
        }
        let checksum = args.next().ok_or("missing_runtime_manifest_checksum")?;
        let yes = match args.next().as_deref() {
            Some("--yes") => true,
            None => false,
            _ => return Err("unknown_argument"),
        };
        if args.next().is_some() {
            return Err("unknown_argument");
        }
        confirm_interruption(yes)?;
        tokio::task::spawn_blocking(move || {
            systemd::install_image_runtime(Path::new(&directory), &checksum)
        })
        .await
        .map_err(|_| "image_runtime_install_task_failed")??;
        println!("kuno_image_runtime_installed");
        return Ok(());
    }
    let mut config = Config::default();
    let mut explicit_config = false;
    let mut service = "ollama.service".to_owned();
    let mut output = None;
    let mut yes = false;
    let mut workers_file = None;
    let mut disable_managed = false;
    let mut response_headers = None;
    let mut stream_idle = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--response-headers" if command == "configure-timeouts" => {
                if response_headers.is_some() {
                    return Err("duplicate_response_header_timeout");
                }
                let seconds = args
                    .next()
                    .ok_or("missing_response_header_timeout")?
                    .parse::<u64>()
                    .map_err(|_| "response_header_timeout_invalid")?;
                config.response_header_timeout_seconds = seconds;
                response_headers = Some(seconds);
            }
            "--stream-idle" if command == "configure-timeouts" => {
                if stream_idle.is_some() {
                    return Err("duplicate_stream_idle_timeout");
                }
                let seconds = args
                    .next()
                    .ok_or("missing_stream_idle_timeout")?
                    .parse::<u64>()
                    .map_err(|_| "stream_idle_timeout_invalid")?;
                config.stream_idle_timeout_seconds = seconds;
                stream_idle = Some(seconds);
            }
            "--workers" if command == "configure-managed" => {
                workers_file = Some(args.next().ok_or("missing_workers_path")?);
            }
            "--disable" if command == "configure-managed" => disable_managed = true,
            "--config"
                if matches!(
                    command.as_str(),
                    "serve"
                        | "resources"
                        | "check"
                        | "install"
                        | "doctor"
                        | "takeover-plan"
                        | "takeover-bundle"
                ) =>
            {
                config = Config::read(Path::new(&args.next().ok_or("missing_config_path")?))?;
                explicit_config = true;
            }
            "--service"
                if matches!(
                    command.as_str(),
                    "takeover-plan" | "takeover-bundle" | "install" | "doctor"
                ) =>
            {
                service = args.next().ok_or("missing_service_name")?
            }
            "--output" if command == "takeover-bundle" => {
                output = Some(args.next().ok_or("missing_output_directory")?)
            }
            "--yes"
                if matches!(
                    command.as_str(),
                    "install"
                        | "upgrade"
                        | "takeover-restore"
                        | "recover"
                        | "configure-managed"
                        | "configure-timeouts"
                ) =>
            {
                yes = true;
            }
            _ => return Err("unknown_argument"),
        }
    }
    config.validate()?;
    match command.as_str() {
        "resources" => {
            if !explicit_config && Path::new("/etc/kuno/config.json").exists() {
                config = Config::read(Path::new("/etc/kuno/config.json"))?;
            }
            let report = kuno::resources::snapshot(config.upstream).await;
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .map_err(|_| "resource_report_encoding_failed")?
            );
            Ok(())
        }
        "check" => {
            println!("configuration_valid");
            Ok(())
        }
        "takeover-plan" => {
            let plan = takeover::inspect(&service, &config)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&plan).map_err(|_| "plan_encoding_failed")?
            );
            Ok(())
        }
        "serve" => serve(config).await,
        "install" => {
            confirm_interruption(yes)?;
            tokio::task::spawn_blocking(move || systemd::install(&service, &config))
                .await
                .map_err(|_| "installation_task_failed")??;
            println!("kuno_installed_and_active");
            Ok(())
        }
        "upgrade" => {
            confirm_interruption(yes)?;
            tokio::task::spawn_blocking(systemd::upgrade)
                .await
                .map_err(|_| "upgrade_task_failed")??;
            println!("kuno_upgraded_and_active");
            Ok(())
        }
        "configure-managed" => {
            if workers_file.is_some() == disable_managed {
                return Err("choose_workers_file_or_disable");
            }
            let pool = if let Some(path) = workers_file {
                if std::fs::metadata(&path)
                    .map_err(|_| "workers_file_unavailable")?
                    .len()
                    > 1024 * 1024
                {
                    return Err("workers_file_too_large");
                }
                Some(
                    serde_json::from_slice(
                        &std::fs::read(path).map_err(|_| "workers_file_unavailable")?,
                    )
                    .map_err(|_| "workers_configuration_invalid")?,
                )
            } else {
                None
            };
            confirm_interruption(yes)?;
            tokio::task::spawn_blocking(move || systemd::configure_managed(pool))
                .await
                .map_err(|_| "managed_configuration_task_failed")??;
            println!("kuno_managed_configuration_active");
            Ok(())
        }
        "configure-timeouts" => {
            if response_headers.is_none() && stream_idle.is_none() {
                return Err("missing_timeout_configuration");
            }
            confirm_interruption(yes)?;
            tokio::task::spawn_blocking(move || {
                systemd::configure_timeouts(response_headers, stream_idle)
            })
            .await
            .map_err(|_| "timeout_configuration_task_failed")??;
            println!("kuno_timeout_configuration_active");
            Ok(())
        }
        "takeover-restore" => {
            confirm_interruption(yes)?;
            tokio::task::spawn_blocking(systemd::restore)
                .await
                .map_err(|_| "restore_task_failed")??;
            println!("original_ollama_service_restored");
            Ok(())
        }
        "doctor" => {
            let report = tokio::task::spawn_blocking(move || systemd::doctor(&service, &config))
                .await
                .map_err(|_| "diagnostic_task_failed")??;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).map_err(|_| "diagnostic_encoding_failed")?
            );
            Ok(())
        }
        "recover" => {
            confirm_interruption(yes)?;
            tokio::task::spawn_blocking(systemd::recover)
                .await
                .map_err(|_| "recovery_task_failed")??;
            println!("managed_upstream_restarted_admission_recovered");
            Ok(())
        }
        "takeover-bundle" => {
            let plan = takeover::inspect(&service, &config)?;
            let executable =
                std::fs::read(std::env::current_exe().map_err(|_| "executable_unavailable")?)
                    .map_err(|_| "executable_unavailable")?;
            takeover::bundle(
                &plan,
                Path::new(&output.ok_or("output_directory_required")?),
                &executable,
            )?;
            println!("takeover_bundle_staged; install_with_kuno_install");
            Ok(())
        }
        _ => Err("unknown_command"),
    }
}

fn confirm_interruption(yes: bool) -> Result<(), &'static str> {
    use std::io::{IsTerminal, Write};
    eprintln!(
        "kuno: This operation briefly interrupts inference. Connected clients may need to retry."
    );
    if yes {
        return Ok(());
    }
    // Never infer consent from a pipe or EOF. Scripts must explicitly pass --yes.
    if !std::io::stdin().is_terminal() {
        return Err("confirmation_required_run_interactively_or_pass_--yes");
    }
    eprint!("Continue? [y/N] ");
    std::io::stderr()
        .flush()
        .map_err(|_| "confirmation_prompt_failed")?;
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|_| "confirmation_read_failed")?;
    if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        Ok(())
    } else {
        Err("cancelled_no_services_changed")
    }
}

async fn serve(mut config: Config) -> Result<(), &'static str> {
    disable_core_dumps()?;
    let token = control_token()?;
    if config.activation_file.is_some() {
        // systemd DynamicUser exposes StateDirectory through a root-controlled
        // symlink. Resolve this one installation path before the ordinary
        // no-symlink state-store check; arbitrary state paths retain that check.
        if config.state_dir == Path::new("/var/lib/kuno") {
            config.state_dir = std::fs::canonicalize(&config.state_dir)
                .map_err(|_| "managed_state_directory_unavailable")?;
        }
        if let Some(pool) = &mut config.managed {
            // The generated unit exposes only the configured shared store at
            // this service-private read-only mount. Keep its host path in the
            // owned config for repeatable upgrades and rollback.
            for worker in &mut pool.workers {
                worker.models = config.state_dir.join("models");
            }
        }
        if let Some(image) = &mut config.image {
            image.models = config.state_dir.join("image-models");
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|_| "startup_health_client_failed")?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
        loop {
            if let Ok(response) = client
                .get(format!("http://{}/api/version", config.upstream))
                .send()
                .await
                && response.status().is_success()
            {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                return Err("upstream_not_ready_at_startup");
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    let gateway = Gateway::persistent(config.clone(), &token)?;
    gateway.initialize_workers().await?;
    if let Some(path) = &config.activation_file {
        gateway.admission.pause(!systemd::activation_ready(path)?);
    }
    // Bind both before accepting traffic, so a control-port collision cannot
    // leave an orphan compatibility listener running without operator access.
    let listener = TcpListener::bind(config.listen)
        .await
        .map_err(|_| "gateway_bind_failed")?;
    let control = TcpListener::bind(config.control_listen)
        .await
        .map_err(|_| "control_bind_failed")?;
    if config.managed.is_some() {
        eprintln!(
            "kuno: managed_experimental; supported inference routes use verified GPU workers; opaque fallback remains unmanaged"
        );
    } else {
        eprintln!("kuno: gateway_only; GPU placement and host memory privacy are not yet verified");
    }
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let shutdown = move || {
        let mut rx = shutdown_rx.clone();
        async move {
            let _ = rx.wait_for(|value| *value).await;
        }
    };
    let server = axum::serve(listener, gateway.compatibility()).with_graceful_shutdown(shutdown());
    let control_server = axum::serve(control, gateway.control()).with_graceful_shutdown(shutdown());
    let both = async {
        tokio::try_join!(server, control_server).map_err(|_| "listener_failed")?;
        Ok(())
    };
    tokio::pin!(both);
    tokio::select! {
        result = &mut both => result,
        _ = shutdown_signal() => {
            gateway.admission.pause(true);
            gateway.stop_images();
            let _ = shutdown_tx.send(true);
            let grace = if config.image.is_some() { 120 } else { 30 };
            let cleanup = async {
                both.await?;
                // A disconnected HTTP request can leave a detached supervisor
                // draining GPU work. Keep the runtime alive until its lease is
                // released, even when both listeners have already drained.
                while gateway.admission.status().active {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                gateway.evict_images().await?;
                Ok(())
            };
            match tokio::time::timeout(Duration::from_secs(grace), cleanup).await {
                Ok(result) => result,
                Err(_) => Err("shutdown_interrupted_upstream_verify_before_restart"),
            }
        }
    }
}

fn control_token() -> Result<String, &'static str> {
    match (
        std::env::var("KUNO_CONTROL_TOKEN").ok(),
        std::env::var_os("KUNO_CONTROL_TOKEN_FILE"),
    ) {
        (Some(token), None) => Ok(token),
        (None, Some(path)) => {
            use std::io::Read;
            let mut token = String::new();
            std::fs::File::open(path)
                .and_then(|f| f.take(257).read_to_string(&mut token))
                .map_err(|_| "control_token_read_failed")?;
            Ok(token)
        }
        (Some(_), Some(_)) => Err("control_token_sources_conflict"),
        _ => Err("control_token_required"),
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        if let Ok(mut terminate) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
        } else {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}

fn disable_core_dumps() -> Result<(), &'static str> {
    #[cfg(unix)]
    {
        let limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: valid pointer to a local rlimit; only this process is affected.
        if unsafe { libc::setrlimit(libc::RLIMIT_CORE, &limit) } != 0 {
            return Err("disable_core_dumps_failed");
        }
    }
    #[cfg(target_os = "linux")]
    // Linux pipe-based crash collectors may ignore RLIMIT_CORE. Prevent dumps
    // at the process level as well. This does not prevent swapping memory.
    if unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) } != 0 {
        return Err("disable_dumpable_failed");
    }
    Ok(())
}
