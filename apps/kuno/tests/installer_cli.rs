use std::process::Command;

#[test]
fn cli_enrollment_validates_user_options_and_never_restarts_services_without_root() {
    let binary = env!("CARGO_BIN_EXE_kuno");
    for (args, error) in [
        (vec!["configure-cli", "--user"], "missing_cli_user"),
        (
            vec!["configure-cli", "--user", "one", "--user", "two"],
            "duplicate_cli_user",
        ),
        (vec!["configure-cli", "--yes"], "unknown_argument"),
        (vec!["upgrade", "--cli-user"], "missing_cli_user"),
        (
            vec!["install", "--cli-user", "one", "--cli-user", "two"],
            "duplicate_cli_user",
        ),
    ] {
        let output = Command::new(binary).args(args).output().unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains(error));
        assert!(!stderr.contains("briefly interrupts inference"));
    }
    if unsafe { libc::geteuid() } != 0 {
        let output = Command::new(binary)
            .args(["configure-cli", "--user", "test-user"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("root_required_run_this_command_with_sudo"));
        assert!(!stderr.contains("briefly interrupts inference"));
    }
}

#[test]
fn installation_requires_root_before_any_mutation() {
    // A root CI environment must never execute an actual installation here.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    for command in ["install", "upgrade", "takeover-restore", "recover"] {
        let output = Command::new(env!("CARGO_BIN_EXE_kuno"))
            .args([command, "--yes"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("briefly interrupts inference"));
        assert!(error.ends_with("kuno: root_required_run_this_command_with_sudo\n"));
    }
}

#[test]
fn activation_requires_an_unchanged_regular_empty_marker() {
    let directory = tempfile::tempdir().unwrap();
    let marker = directory.path().join("ready");
    assert!(!kuno::systemd::activation_ready(&marker).unwrap());
    std::fs::write(&marker, []).unwrap();
    assert!(kuno::systemd::activation_ready(&marker).unwrap());
    std::fs::write(&marker, "administrator edit").unwrap();
    assert!(kuno::systemd::activation_ready(&marker).is_err());
    #[cfg(unix)]
    {
        let link = directory.path().join("link");
        std::os::unix::fs::symlink(marker, &link).unwrap();
        assert!(kuno::systemd::activation_ready(&link).is_err());
    }
}

#[test]
fn image_configuration_requires_confirmation_before_privileged_work() {
    let output = Command::new(env!("CARGO_BIN_EXE_kuno"))
        .args(["configure-image", "--disable"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("confirmation_required"));
    if unsafe { libc::geteuid() } != 0 {
        let output = Command::new(env!("CARGO_BIN_EXE_kuno"))
            .args(["configure-image", "--disable", "--yes"])
            .output()
            .unwrap();
        assert!(String::from_utf8_lossy(&output.stderr).contains("root_required"));
    }
}

#[test]
fn optional_runtime_install_requires_confirmation_and_root_before_bundle_access() {
    let mut args = vec![
        "install-image-runtime",
        "--directory",
        "/nonexistent/public-runtime-bundle",
        "--manifest-sha256",
    ];
    let digest = "0".repeat(64);
    args.push(&digest);
    let output = Command::new(env!("CARGO_BIN_EXE_kuno"))
        .args(&args)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("confirmation_required"));
    if unsafe { libc::geteuid() } != 0 {
        args.push("--yes");
        let output = Command::new(env!("CARGO_BIN_EXE_kuno"))
            .args(&args)
            .output()
            .unwrap();
        assert!(String::from_utf8_lossy(&output.stderr).contains("root_required"));
    }
}

#[test]
fn unattended_commands_require_explicit_yes_before_mutation() {
    for command in ["install", "upgrade", "takeover-restore", "recover"] {
        let output = Command::new(env!("CARGO_BIN_EXE_kuno"))
            .arg(command)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("briefly interrupts inference"));
        assert!(error.contains("confirmation_required_run_interactively_or_pass_--yes"));
    }
}

#[test]
fn interactive_commands_wait_and_default_to_cancellation() {
    use std::{
        fs::File,
        io::Write,
        os::fd::FromRawFd,
        process::Stdio,
        time::{Duration, Instant},
    };
    for command in ["install", "upgrade", "takeover-restore", "recover"] {
        for answer in ["\n", "n\n", "yes\n"] {
            // Confirmed root commands would operate on the real host.
            if answer == "yes\n" && unsafe { libc::geteuid() } == 0 {
                continue;
            }
            let (mut master, mut slave) = (-1, -1);
            assert_eq!(
                unsafe {
                    libc::openpty(
                        &mut master,
                        &mut slave,
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        std::ptr::null(),
                    )
                },
                0
            );
            let mut terminal = unsafe { File::from_raw_fd(master) };
            let input = unsafe { File::from_raw_fd(slave) };
            let output = tempfile::NamedTempFile::new().unwrap();
            let mut child = Command::new(env!("CARGO_BIN_EXE_kuno"))
                .arg(command)
                .stdin(Stdio::from(input))
                .stdout(Stdio::null())
                .stderr(output.reopen().unwrap())
                .spawn()
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while !std::fs::read_to_string(output.path())
                .unwrap()
                .contains("Continue? [y/N]")
            {
                if Instant::now() > deadline || child.try_wait().unwrap().is_some() {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("confirmation prompt missing");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(
                child.try_wait().unwrap().is_none(),
                "must wait before any action"
            );
            terminal.write_all(answer.as_bytes()).unwrap();
            while child.try_wait().unwrap().is_none() {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("confirmation did not finish");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let error = std::fs::read_to_string(output.path()).unwrap();
            assert_eq!(error.matches("Continue? [y/N]").count(), 1);
            assert!(
                error.contains(if answer == "yes\n" {
                    "root_required_run_this_command_with_sudo"
                } else {
                    "cancelled_no_services_changed"
                }),
                "{error}"
            );
        }
    }
}

#[test]
fn managed_configuration_requires_explicit_mode_confirmation_and_root() {
    let binary = env!("CARGO_BIN_EXE_kuno");
    let output = Command::new(binary)
        .arg("configure-managed")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("choose_workers_file_or_disable")
    );
    let output = Command::new(binary)
        .args(["configure-managed", "--disable"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("confirmation_required")
    );
    if unsafe { libc::geteuid() } != 0 {
        let output = Command::new(binary)
            .args(["configure-managed", "--disable", "--yes"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8(output.stderr)
                .unwrap()
                .contains("root_required_run_this_command_with_sudo")
        );
    }
    // Never silently ignore a configuration supplied to an upgrade.
    let output = Command::new(binary)
        .args(["upgrade", "--config", "unused"])
        .output()
        .unwrap();
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("unknown_argument")
    );
}

#[test]
fn automatic_recovery_is_root_only_and_takes_no_user_arguments() {
    let bad = Command::new(env!("CARGO_BIN_EXE_kuno"))
        .args(["auto-recover", "--yes"])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("unknown_argument"));
    if unsafe { libc::geteuid() } != 0 {
        let output = Command::new(env!("CARGO_BIN_EXE_kuno"))
            .arg("auto-recover")
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("root_required"));
    }
}

#[test]
fn timeout_configuration_validates_arguments_before_confirmation_or_privileged_work() {
    let binary = env!("CARGO_BIN_EXE_kuno");
    for (args, expected) in [
        (vec![], "missing_timeout_configuration"),
        (
            vec!["--response-headers"],
            "missing_response_header_timeout",
        ),
        (vec!["--stream-idle"], "missing_stream_idle_timeout"),
        (
            vec!["--response-headers", "abc"],
            "response_header_timeout_invalid",
        ),
        (vec!["--stream-idle", "-1"], "stream_idle_timeout_invalid"),
        (
            vec!["--response-headers", "86401"],
            "invalid_admission_bounds",
        ),
        (vec!["--stream-idle", "86401"], "invalid_admission_bounds"),
        (
            vec!["--response-headers", "0", "--response-headers", "300"],
            "duplicate_response_header_timeout",
        ),
        (
            vec!["--stream-idle", "0", "--stream-idle", "300"],
            "duplicate_stream_idle_timeout",
        ),
        (vec!["--config", "unused"], "unknown_argument"),
    ] {
        let output = Command::new(binary)
            .arg("configure-timeouts")
            .args(&args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains(expected), "{args:?}: {error}");
        assert!(!error.contains("briefly interrupts inference"));
    }
    for args in [
        vec!["--response-headers", "0"],
        vec!["--stream-idle", "0"],
        vec!["--response-headers", "0", "--stream-idle", "0"],
        vec!["--stream-idle", "86400", "--response-headers", "86400"],
    ] {
        let output = Command::new(binary)
            .arg("configure-timeouts")
            .args(&args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("confirmation_required"));
        if unsafe { libc::geteuid() } != 0 {
            let output = Command::new(binary)
                .arg("configure-timeouts")
                .args(&args)
                .arg("--yes")
                .output()
                .unwrap();
            assert!(!output.status.success());
            assert!(String::from_utf8_lossy(&output.stderr).contains("root_required"));
        }
    }
    let output = Command::new(binary)
        .args(["upgrade", "--response-headers", "0"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown_argument"));
}

#[test]
fn native_configuration_requires_confirmation_before_privileged_work() {
    let binary = env!("CARGO_BIN_EXE_kuno");
    let output = Command::new(binary)
        .args(["configure-native-text", "--disable"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("confirmation_required"));
    if unsafe { libc::geteuid() } != 0 {
        let output = Command::new(binary)
            .args(["configure-native-text", "--disable", "--yes"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("root_required"));
    }
}
