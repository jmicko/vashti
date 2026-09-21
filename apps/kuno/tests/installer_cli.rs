use std::process::Command;

#[test]
fn mutating_commands_require_an_explicit_maintenance_window() {
    for command in ["install", "takeover-restore", "recover"] {
        let output = Command::new(env!("CARGO_BIN_EXE_kuno"))
            .arg(command)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(
            error.contains("root_required")
                || error.contains("close_inference_clients_then_pass_quiesced"),
            "{error}"
        );
    }
}

#[test]
fn installation_requires_root_before_any_mutation() {
    // A root CI environment must never execute an actual installation here.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    for command in ["install", "takeover-restore", "recover"] {
        let output = Command::new(env!("CARGO_BIN_EXE_kuno"))
            .args([command, "--quiesced"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            "kuno: root_required_run_this_command_with_sudo\n"
        );
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
