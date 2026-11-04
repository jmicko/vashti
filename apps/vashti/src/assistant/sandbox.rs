use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use serde::Serialize;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::Command,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::service;
use crate::{app_state::AppState, auth::service::unix_timestamp, error::ApiError};

pub const IMAGE: &str = "docker.io/library/python:3.12-slim-bookworm";
const ARCHIVE_LIMIT: usize = 64 * 1024 * 1024;
pub const ARTIFACT_LIMIT: usize = 20 * 1024 * 1024;

#[derive(Clone, Debug, Serialize)]
pub struct Availability {
    pub available: bool,
    pub image: String,
    pub reason: Option<String>,
    #[serde(skip)]
    pub image_id: Option<String>,
}

struct Captured {
    bytes: Vec<u8>,
    stderr: Vec<u8>,
    code: i32,
    truncated: bool,
}

async fn read_bounded(
    mut reader: impl AsyncRead + Unpin,
    limit: usize,
) -> std::io::Result<(Vec<u8>, bool)> {
    let mut result = Vec::new();
    let mut truncated = false;
    let mut buf = [0u8; 8192];
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        let take = n.min(limit.saturating_sub(result.len()));
        result.extend_from_slice(&buf[..take]);
        truncated |= take < n;
    }
    Ok((result, truncated))
}

async fn command(
    args: &[String],
    input: Option<&[u8]>,
    limit: usize,
    seconds: u64,
    cancel: &CancellationToken,
) -> Result<Captured, ApiError> {
    let mut child = Command::new("podman")
        .args(args)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| {
            ApiError::bad_request(
                "sandbox_unavailable",
                "Rootless Podman is unavailable on this server",
            )
        })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ApiError::internal("Sandbox output unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ApiError::internal("Sandbox output unavailable"))?;
    let mut stdin = child.stdin.take();
    let work = async {
        let write = async {
            if let (Some(mut writer), Some(bytes)) = (stdin.take(), input) {
                writer.write_all(bytes).await?;
                writer.shutdown().await?;
                drop(writer);
            }
            Ok::<(), std::io::Error>(())
        };
        let (out, err, status, _) = tokio::try_join!(
            read_bounded(stdout, limit),
            read_bounded(stderr, 8192),
            child.wait(),
            write
        )?;
        Ok::<Captured, std::io::Error>(Captured {
            bytes: out.0,
            stderr: err.0,
            code: status.code().unwrap_or(-1),
            truncated: out.1 || err.1,
        })
    };
    tokio::select! {
        _=cancel.cancelled()=>Err(ApiError::conflict("sandbox_cancelled","Sandbox operation cancelled")),
        outcome=tokio::time::timeout(Duration::from_secs(seconds),work)=>match outcome {
            Ok(Ok(value))=>Ok(value),
            Ok(Err(_))=>Err(ApiError::internal("Sandbox process failed")),
            Err(_)=>Err(ApiError::bad_request("sandbox_timeout","Sandbox operation exceeded its time limit")),
        }
    }
}

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| s.to_string()).collect()
}

pub async fn availability() -> Availability {
    let mut result = Availability {
        available: false,
        image: IMAGE.into(),
        image_id: None,
        reason: None,
    };
    let token = CancellationToken::new();
    let info = command(
        &args(&["info", "--format", "json"]),
        None,
        128 * 1024,
        10,
        &token,
    )
    .await;
    let rootless = info
        .ok()
        .filter(|v| v.code == 0)
        .and_then(|v| serde_json::from_slice::<Value>(&v.bytes).ok())
        .and_then(|v| {
            v.pointer("/host/security/rootless")
                .and_then(Value::as_bool)
        });
    if rootless != Some(true) {
        result.reason =
            Some("Rootless Podman is required. No privileged or rootful fallback is used.".into());
        return result;
    }
    let image = command(
        &args(&["image", "inspect", "--format", "{{.Id}}", IMAGE]),
        None,
        1024,
        10,
        &token,
    )
    .await;
    let id = image.ok().filter(|v| v.code == 0).map(|v| {
        String::from_utf8_lossy(&v.bytes)
            .trim()
            .trim_start_matches("sha256:")
            .to_string()
    });
    if let Some(id) = id.filter(|id| id.len() == 64 && id.bytes().all(|c| c.is_ascii_hexdigit())) {
        result.available = true;
        result.image_id = Some(id);
    } else {
        result.reason = Some(format!(
            "The sandbox image is not installed. A server administrator can run: podman pull {IMAGE}"
        ));
    }
    result
}

fn storage(state: &AppState) -> PathBuf {
    state.config.data_dir.join("assistant")
}

fn run_key(run: &str) -> Result<(), ApiError> {
    Uuid::parse_str(run)
        .map(|_| ())
        .map_err(|_| ApiError::bad_request("invalid_run", "Invalid run identifier"))
}

pub fn container_args(name: &str, image: &str, installation: &str) -> Vec<String> {
    args(&[
        "run",
        "--detach",
        "--pull=never",
        "--name",
        name,
        "--label",
        &format!("chat.vashti.assistant={installation}"),
        "--network=none",
        "--http-proxy=false",
        "--pid=private",
        "--ipc=private",
        "--uts=private",
        "--no-hosts",
        "--read-only",
        "--read-only-tmpfs=false",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=128",
        "--memory=1g",
        "--memory-swap=1g",
        "--cpus=1",
        "--ulimit=nofile=256:256",
        "--tmpfs=/workspace:rw,nosuid,nodev,size=512m,mode=700",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777",
        "--workdir=/workspace",
        "--env=HOME=/workspace",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--log-driver=none",
        "--entrypoint=/bin/sleep",
        image,
        "infinity",
    ])
}

async fn remove(name: &str) {
    let _ = command(
        &args(&["rm", "--force", "--time=1", name]),
        None,
        1024,
        10,
        &CancellationToken::new(),
    )
    .await;
}

async fn create(
    state: &AppState,
    run: &str,
    image: &str,
    token: &CancellationToken,
) -> Result<String, ApiError> {
    run_key(run)?;
    if image.len() != 64 || !image.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(ApiError::internal("Stored sandbox image is invalid"));
    }
    let name = format!("vashti-assistant-{}-{}", &run[..8], Uuid::new_v4().simple());
    let result = command(
        &container_args(&name, image, &state.server_instance_id),
        None,
        1024,
        30,
        token,
    )
    .await;
    if !result.as_ref().is_ok_and(|v| v.code == 0) {
        remove(&name).await;
        return Err(ApiError::bad_request(
            "sandbox_start_failed",
            "Podman could not start the isolated workspace with the required resource/security limits",
        ));
    }
    let inspected = command(
        &args(&["inspect", "--format", "{{json .}}", &name]),
        None,
        256 * 1024,
        10,
        token,
    )
    .await;
    let safe = inspected
        .ok()
        .filter(|v| v.code == 0 && !v.truncated)
        .and_then(|v| serde_json::from_slice::<Value>(&v.bytes).ok())
        .is_some_and(|v| {
            v.pointer("/HostConfig/NetworkMode").and_then(Value::as_str) == Some("none")
                && v.pointer("/HostConfig/ReadonlyRootfs")
                    .and_then(Value::as_bool)
                    == Some(true)
                && v.pointer("/HostConfig/Privileged").and_then(Value::as_bool) == Some(false)
                && v.get("Mounts")
                    .and_then(Value::as_array)
                    .is_some_and(|mounts| {
                        mounts.iter().all(|m| {
                            m.get("Type").and_then(Value::as_str) == Some("tmpfs")
                                && m.get("Destination")
                                    .and_then(Value::as_str)
                                    .is_some_and(|d| matches!(d, "/workspace" | "/tmp"))
                        })
                    })
        });
    if !safe {
        remove(&name).await;
        return Err(ApiError::forbidden(
            "unsafe_sandbox_config",
            "Podman configuration introduced unexpected mounts or weakened the required isolation. The workspace was not used.",
        ));
    }
    let archive = storage(state)
        .join("workspaces")
        .join(format!("{run}.tar.gz"));
    match tokio::fs::read(&archive).await {
        Ok(bytes) if bytes.len() <= ARCHIVE_LIMIT => {
            let restored = command(
                &args(&[
                    "exec",
                    "-i",
                    &name,
                    "tar",
                    "-xzf",
                    "-",
                    "--no-same-owner",
                    "--no-same-permissions",
                    "-C",
                    "/workspace",
                ]),
                Some(&bytes),
                1024,
                30,
                token,
            )
            .await;
            if !restored.as_ref().is_ok_and(|v| v.code == 0) {
                remove(&name).await;
                return Err(ApiError::internal("Could not restore the saved workspace"));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => {
            remove(&name).await;
            return Err(ApiError::internal(
                "Saved workspace is invalid or too large",
            ));
        }
    }
    Ok(name)
}

async fn private_write(path: &Path, bytes: &[u8]) -> Result<(), ApiError> {
    let parent = path
        .parent()
        .ok_or_else(|| ApiError::internal("Invalid artifact destination"))?;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|_| ApiError::internal("Could not create assistant storage"))?;
    let temp = parent.join(format!("{}.tmp", Uuid::new_v4()));
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options
        .open(&temp)
        .await
        .map_err(|_| ApiError::internal("Could not create assistant file"))?;
    let result = async {
        file.write_all(bytes).await?;
        file.sync_all().await?;
        tokio::fs::rename(&temp, path).await
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(ApiError::internal("Could not save assistant file"));
    }
    Ok(())
}

pub async fn execute(
    state: &AppState,
    run: &str,
    image: &str,
    command_text: &str,
    timeout: u64,
    cancel: &CancellationToken,
) -> Result<Value, ApiError> {
    if command_text.trim().is_empty()
        || command_text.len() > 32_000
        || !(1..=300).contains(&timeout)
    {
        return Err(ApiError::bad_request(
            "invalid_command",
            "Use a nonempty command up to 32000 bytes and a 1-300 second limit",
        ));
    }
    let name = create(state, run, image, cancel).await?;
    let work=async {
        let output=command(&args(&["exec",&name,"timeout","--signal=TERM","--kill-after=2s",&timeout.to_string(),"/bin/sh","-c",command_text]),None,48*1024,timeout+10,cancel).await?;
        // Only tar bytes leave the container; restoration also happens inside a
        // new container. Host paths and archive members are never unpacked here.
        let archive=command(&args(&["exec",&name,"tar","-czf","-","-C","/workspace","."]),None,ARCHIVE_LIMIT,45,cancel).await?;
        if archive.code!=0 || archive.truncated { return Err(ApiError::bad_request("workspace_too_large","Workspace could not be checkpointed within the 64 MiB archive limit")); }
        private_write(&storage(state).join("workspaces").join(format!("{run}.tar.gz")),&archive.bytes).await?;
        Ok(json!({"exit_code":output.code,"stdout":String::from_utf8_lossy(&output.bytes),"stderr":String::from_utf8_lossy(&output.stderr),"output_truncated":output.truncated,"timed_out":output.code==124||output.code==137,"workspace":"/workspace","network":"disabled"}))
    }.await;
    remove(&name).await;
    work
}

pub async fn export(
    state: &AppState,
    run: &str,
    image: &str,
    path: &str,
    cancel: &CancellationToken,
) -> Result<Value, ApiError> {
    if path.len() > 1024 || path.contains('\0') {
        return Err(ApiError::bad_request(
            "invalid_path",
            "Invalid workspace file path",
        ));
    }
    let name = create(state, run, image, cancel).await?;
    let script = "import pathlib,sys,stat\np=(pathlib.Path('/workspace')/sys.argv[1]).resolve(strict=True)\nif not p.is_relative_to('/workspace'): raise SystemExit(2)\nf=p.open('rb');s=__import__('os').fstat(f.fileno())\nif not stat.S_ISREG(s.st_mode) or s.st_size>20971520: raise SystemExit(3)\nsys.stdout.buffer.write(f.read(20971521))";
    let result = command(
        &args(&["exec", &name, "python3", "-I", "-c", script, path]),
        None,
        ARTIFACT_LIMIT + 1,
        30,
        cancel,
    )
    .await;
    remove(&name).await;
    let result = result?;
    if result.code != 0 || result.bytes.len() > ARTIFACT_LIMIT || result.truncated {
        return Err(ApiError::bad_request(
            "invalid_artifact",
            "Choose a regular file inside /workspace, at most 20 MiB",
        ));
    }
    let (count, run_bytes): (i64, i64) = sqlx::query_as(
        "SELECT count(*),COALESCE(sum(size),0) FROM assistant_artifacts WHERE run_id=?",
    )
    .bind(run)
    .fetch_one(&state.db)
    .await?;
    let user_bytes:i64=sqlx::query_scalar("SELECT COALESCE(sum(a.size),0) FROM assistant_artifacts a JOIN assistant_runs r ON r.id=a.run_id WHERE r.user_id=(SELECT user_id FROM assistant_runs WHERE id=?)")
        .bind(run).fetch_one(&state.db).await?;
    if count >= 32
        || run_bytes + result.bytes.len() as i64 > 128 * 1024 * 1024
        || user_bytes + result.bytes.len() as i64 > 512 * 1024 * 1024
    {
        return Err(ApiError::bad_request(
            "artifact_quota",
            "Artifact storage is full (32 files/128 MiB per run, 512 MiB per account). The owner can download and delete older artifacts.",
        ));
    }
    let filename = Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("output.bin");
    let filename: String = filename
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ' '))
        .take(120)
        .collect();
    let filename = if filename.trim_matches('.').trim().is_empty() {
        "output.bin".to_string()
    } else {
        filename
    };
    let id = Uuid::new_v4().to_string();
    let target = storage(state).join("artifacts").join(&id);
    private_write(&target, &result.bytes).await?;
    let mime = mime_guess::from_path(&filename)
        .first_or_octet_stream()
        .to_string();
    let inserted=sqlx::query("INSERT INTO assistant_artifacts(id,run_id,name,mime_type,size,created_at) VALUES(?,?,?,?,?,?)")
        .bind(&id).bind(run).bind(&filename).bind(&mime).bind(result.bytes.len() as i64).bind(unix_timestamp()).execute(&state.db).await;
    if let Err(e) = inserted {
        let _ = tokio::fs::remove_file(target).await;
        return Err(e.into());
    }
    service::append(&state.db, run, "artifact", &filename, json!({"id":id})).await?;
    Ok(
        json!({"artifact_id":id,"name":filename,"size":result.bytes.len(),"download_url":format!("/api/assistant/artifacts/{id}")}),
    )
}

pub fn artifact_path(state: &AppState, id: &str) -> Result<PathBuf, ApiError> {
    run_key(id)?;
    Ok(storage(state).join("artifacts").join(id))
}

pub async fn recover(state: &AppState) {
    let selector = format!("label=chat.vashti.assistant={}", state.server_instance_id);
    if let Ok(list) = command(
        &args(&[
            "ps",
            "--all",
            "--filter",
            &selector,
            "--format",
            "{{.Names}}",
        ]),
        None,
        32 * 1024,
        10,
        &CancellationToken::new(),
    )
    .await
        && list.code == 0
    {
        for name in String::from_utf8_lossy(&list.bytes)
            .lines()
            .filter(|name| name.starts_with("vashti-assistant-"))
        {
            remove(name).await;
        }
    }
    if let Ok(mut entries) = tokio::fs::read_dir(storage(state).join("artifacts")).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let id = entry.file_name().to_string_lossy().to_string();
            if run_key(&id).is_err() {
                continue;
            }
            let exists: Result<bool, _> =
                sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM assistant_artifacts WHERE id=?)")
                    .bind(&id)
                    .fetch_one(&state.db)
                    .await;
            if matches!(exists, Ok(false)) {
                let _ = tokio::fs::remove_file(entry.path()).await;
            }
        }
    }
}
