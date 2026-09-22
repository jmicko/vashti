use std::process::Command;

#[test]
fn installation_requires_root_before_any_mutation() {
    // A root CI environment must never execute an actual installation here.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    for command in ["install", "takeover-restore", "recover"] {
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
fn unattended_commands_require_explicit_yes_before_mutation() {
    for command in ["install", "takeover-restore", "recover"] {
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
    for command in ["install", "takeover-restore", "recover"] {
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
