use kuno::{config::Config, gateway::Gateway, systemd, takeover};
use std::{path::Path, time::Duration};
use tokio::net::TcpListener;

const HELP: &str = "Kuno — local inference gateway (development)\n\n\
Usage:\n  kuno serve [--config FILE]\n  kuno check [--config FILE]\n  kuno doctor [--service ollama.service]\n  kuno install --quiesced [--service ollama.service]\n  kuno recover --quiesced\n  kuno takeover-restore --quiesced\n  kuno takeover-plan [--config FILE] [--service ollama.service]\n  kuno takeover-bundle --output NEW_DIRECTORY [--config FILE] [--service ollama.service]\n\n\
serve requires KUNO_CONTROL_TOKEN or KUNO_CONTROL_TOKEN_FILE (32–256 URL-safe characters).\n\
Default gateway: 127.0.0.1:11433; control: 127.0.0.1:11432; upstream: 127.0.0.1:11434.\n\
takeover-plan is read-only. It never changes services, ports, or model limits.\n";

#[tokio::main]
async fn main() {
    if let Err(code) = run().await {
        // Only fixed error codes reach diagnostics. Never emit backend errors,
        // request URLs, configuration contents, tokens, or inference payloads.
        eprintln!("kuno: {code}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), &'static str> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".into());
    if matches!(command.as_str(), "help" | "--help" | "-h") {
        print!("{HELP}");
        return Ok(());
    }
    let mut config = Config::default();
    let mut service = "ollama.service".to_owned();
    let mut output = None;
    let mut quiesced = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                config = Config::read(Path::new(&args.next().ok_or("missing_config_path")?))?
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
            "--quiesced"
                if matches!(command.as_str(), "install" | "takeover-restore" | "recover") =>
            {
                quiesced = true
            }
            _ => return Err("unknown_argument"),
        }
    }
    config.validate()?;
    match command.as_str() {
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
            tokio::task::spawn_blocking(move || systemd::install(&service, &config, quiesced))
                .await
                .map_err(|_| "installation_task_failed")??;
            println!("kuno_installed_and_active");
            Ok(())
        }
        "takeover-restore" => {
            tokio::task::spawn_blocking(move || systemd::restore(quiesced))
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
            tokio::task::spawn_blocking(move || systemd::recover(quiesced))
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
            println!("takeover_bundle_staged; install_with_kuno_install_quiesced");
            Ok(())
        }
        _ => Err("unknown_command"),
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
    eprintln!("kuno: gateway_only; GPU placement and host memory privacy are not yet verified");
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
            let _ = shutdown_tx.send(true);
            match tokio::time::timeout(Duration::from_secs(30), both).await {
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
