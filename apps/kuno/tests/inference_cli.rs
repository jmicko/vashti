use std::{
    fs,
    io::Write,
    os::unix::{fs::PermissionsExt, process::ExitStatusExt},
    process::{Command, Stdio},
};

fn fixture(directory: &std::path::Path, body: &str) {
    let path = directory.join("ollama");
    fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn command(directory: &std::path::Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kuno"));
    command
        .env("PATH", directory)
        .env_remove("KUNO_HOST")
        .env("OLLAMA_HOST", "http://127.0.0.1:19999")
        .env("OLLAMA_NOHISTORY", "0");
    command
}

#[test]
fn subprocess_targets_kuno_preserves_arguments_and_inherits_io() {
    let directory = tempfile::tempdir().unwrap();
    fixture(
        directory.path(),
        "printf '%s\\n' \"$OLLAMA_HOST\" \"$OLLAMA_NOHISTORY\"; printf '%s\\0' \"$@\"; /bin/cat; exit 37",
    );
    let mut child = command(directory.path())
        .env("KUNO_HOST", "http://0.0.0.0:12000")
        .args([
            "run",
            "fixture",
            "spaces\nquotes' $literal",
            "--config",
            "unchanged",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"synthetic stdin")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(37));
    assert_eq!(output.stdout, b"http://127.0.0.1:12000\n1\nrun\0fixture\0spaces\nquotes' $literal\0--config\0unchanged\0synthetic stdin");
    assert!(output.stderr.is_empty());
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn explicit_host_beats_both_environment_variables_and_config() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), "printf '%s\\n' \"$OLLAMA_HOST\"");
    let output = command(directory.path())
        .env("KUNO_HOST", "http://127.0.0.1:12000")
        .args([
            "--config",
            "/missing/config.json",
            "--host",
            "[::]:12001",
            "ps",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, b"http://[::1]:12001\n");
}

#[test]
fn original_ollama_host_is_ignored_and_selected_config_is_used() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), "printf '%s\\n' \"$OLLAMA_HOST\"");
    let config = directory.path().join("config.json");
    fs::write(&config, r#"{"listen":"0.0.0.0:12002"}"#).unwrap();
    let output = command(directory.path())
        .arg("--config")
        .arg(&config)
        .arg("list")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, b"http://127.0.0.1:12002\n");
}

#[test]
fn invalid_endpoint_errors_are_content_free_and_do_not_start_the_client() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), "echo unexpected_client_start");
    let output = command(directory.path())
        .args([
            "--host",
            "http://user:synthetic-secret@localhost:12003",
            "run",
            "fixture",
            "synthetic prompt",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(output.stderr, b"kuno: cli_endpoint_invalid\n");
}

#[test]
fn exec_preserves_client_signal_exit_instead_of_interpreting_it_as_runtime_error() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path(), "kill -TERM $$");
    let output = command(directory.path())
        .args(["--host", "localhost:12004", "ps"])
        .output()
        .unwrap();
    assert_eq!(output.status.signal(), Some(libc::SIGTERM));
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}

#[test]
fn recursive_client_is_rejected_before_exec() {
    let directory = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_kuno"), directory.path().join("ollama"))
        .unwrap();
    let output = command(directory.path())
        .args(["--host", "localhost:12005", "ps"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stderr, b"kuno: ollama_cli_cannot_be_kuno\n");
}
