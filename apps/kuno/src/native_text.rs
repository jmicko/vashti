//! Opt-in native text adapter. A distinct owned name leaves existing Ollama
//! names and opaque proxy behavior intact. The child holds model-only residency.
use crate::native_protocol as wire;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    process::Stdio,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Mutex, mpsc},
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, &'static str>;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model: String,
    pub executable: PathBuf,
    pub settings: PathBuf,
    pub pci_address: String,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if !self.model.starts_with("kuno-")
            || self.model.len() > 128
            || !self
                .model
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
            || !self.executable.is_absolute()
            || !self.settings.is_absolute()
            || self.pci_address.len() != 12
        {
            return Err("native_text_configuration_invalid");
        }
        Ok(())
    }
    pub fn metadata(&self) -> Value {
        json!({"name":self.model,"model":self.model,"size":83617662656u64,
            "digest":"4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3","modified_at":"2026-10-19T00:00:00Z",
            "details":{"format":"gguf","family":"qwen4exp","families":["qwen4exp"],"quantization_level":"IQ3_S"},"kuno_backend":"rust-intel"})
    }
    pub fn show(&self) -> Value {
        json!({"license":"See model artifact license","parameters":"temperature 0\nnum_ctx 32768\nnum_predict 1024",
            "template":"{{ .Prompt }}", "capabilities":["completion"],
            "details":self.metadata()["details"],"model_info":{"general.architecture":"qwen4exp","qwen4exp.context_length":32768},"kuno_backend":"rust-intel"})
    }
}
pub struct Prepared {
    pub request: Option<wire::Generate>,
    pub chat: bool,
    pub stream: bool,
    pub keep_alive: bool,
}
fn preload() -> wire::Generate {
    wire::Generate {
        prompt: Zeroizing::new(Vec::new()),
        outputs: 0,
        seed: 0,
        temperature: 0.,
        top_p: 1.,
        min_p: 0.,
        top_k: 1,
    }
}
fn append(out: &mut Vec<u8>, text: &str) -> Result<()> {
    if out
        .len()
        .checked_add(text.len())
        .is_none_or(|n| n > wire::MAX_PROMPT_BYTES)
    {
        return Err("native_prompt_too_large");
    }
    out.extend_from_slice(text.as_bytes());
    Ok(())
}
/// Validate before dispatch. No silently ignored tools, images or output format.
/// This preview implements non-thinking text chat and rendered/raw generation.
pub fn prepare(path: &str, value: &Value) -> Result<Prepared> {
    let object = value.as_object().ok_or("native_request_invalid")?;
    let chat = match path {
        "/api/chat" => true,
        "/api/generate" => false,
        _ => return Err("native_endpoint_unsupported"),
    };
    let allowed = if chat {
        &[
            "model",
            "messages",
            "stream",
            "options",
            "keep_alive",
            "think",
        ][..]
    } else {
        &[
            "model",
            "prompt",
            "system",
            "raw",
            "stream",
            "options",
            "keep_alive",
            "think",
        ][..]
    };
    if object.keys().any(|k| !allowed.contains(&k.as_str())) {
        return Err("native_request_feature_unsupported");
    }
    if value.get("think").is_some_and(|v| v != false) {
        return Err("native_thinking_unsupported");
    }
    let stream = match value.get("stream") {
        None => true,
        Some(v) => v.as_bool().ok_or("native_request_invalid")?,
    };
    let keep_alive = match value.get("keep_alive") {
        None => true,
        Some(v) if v == 0 || v == "0" || v == "0s" => false,
        Some(v) if v == -1 || v == "-1" => true,
        _ => return Err("native_keep_alive_unsupported"),
    };
    let options = match value.get("options") {
        None => None,
        Some(v) => Some(v.as_object().ok_or("native_options_invalid")?),
    };
    if options.is_some_and(|o| {
        o.keys().any(|k| {
            ![
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "seed",
                "num_predict",
                "num_ctx",
            ]
            .contains(&k.as_str())
        })
    }) {
        return Err("native_option_unsupported");
    }
    let get = |name: &str| options.and_then(|o| o.get(name));
    if get("num_ctx").is_some_and(|v| v.as_u64() != Some(32768)) {
        return Err("native_context_option_unsupported");
    }
    let float = |name, default| -> Result<f32> {
        match get(name) {
            None => Ok(default),
            Some(v) => v.as_f64().map(|f| f as f32).ok_or("native_options_invalid"),
        }
    };
    let outputs = match get("num_predict") {
        None => 1024,
        Some(v) => u32::try_from(v.as_u64().ok_or("native_options_invalid")?)
            .map_err(|_| "native_options_invalid")?,
    };
    let top_k = match get("top_k") {
        None => 20,
        Some(v) => i32::try_from(v.as_i64().ok_or("native_options_invalid")?)
            .map_err(|_| "native_options_invalid")?,
    };
    let seed = match get("seed") {
        None => 0,
        Some(v) => v.as_u64().ok_or("native_options_invalid")?,
    };
    let mut prompt = Zeroizing::new(Vec::with_capacity(wire::MAX_PROMPT_BYTES));
    if chat {
        let messages = value
            .get("messages")
            .and_then(Value::as_array)
            .ok_or("native_messages_invalid")?;
        if messages.is_empty() {
            return Ok(Prepared {
                request: if keep_alive { Some(preload()) } else { None },
                chat,
                stream,
                keep_alive,
            });
        }
        for message in messages {
            let m = message.as_object().ok_or("native_messages_invalid")?;
            if m.keys().any(|k| !["role", "content"].contains(&k.as_str())) {
                return Err("native_message_feature_unsupported");
            }
            let role = m
                .get("role")
                .and_then(Value::as_str)
                .ok_or("native_messages_invalid")?;
            if !["user", "system", "assistant"].contains(&role) {
                return Err("native_message_role_unsupported");
            }
            let content = m
                .get("content")
                .and_then(Value::as_str)
                .ok_or("native_messages_invalid")?;
            append(&mut prompt, "<|im_start|>")?;
            append(&mut prompt, role)?;
            append(&mut prompt, "\n")?;
            append(&mut prompt, content)?;
            append(&mut prompt, "<|im_end|>\n")?;
        }
    } else {
        let input = value.get("prompt").and_then(Value::as_str).unwrap_or("");
        if input.is_empty() && value.get("system").is_none() {
            return Ok(Prepared {
                request: if keep_alive { Some(preload()) } else { None },
                chat,
                stream,
                keep_alive,
            });
        }
        let raw = match value.get("raw") {
            None => false,
            Some(v) => v.as_bool().ok_or("native_request_invalid")?,
        };
        if raw {
            if value.get("system").is_some() {
                return Err("native_raw_system_unsupported");
            }
            append(&mut prompt, input)?;
        } else {
            if let Some(system) = value.get("system") {
                let system = system.as_str().ok_or("native_request_invalid")?;
                append(&mut prompt, "<|im_start|>system\n")?;
                append(&mut prompt, system)?;
                append(&mut prompt, "<|im_end|>\n")?;
            }
            append(&mut prompt, "<|im_start|>user\n")?;
            append(&mut prompt, input)?;
            append(&mut prompt, "<|im_end|>\n")?;
        }
    }
    if chat || value.get("raw").and_then(Value::as_bool) != Some(true) {
        append(
            &mut prompt,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )?;
    }
    let request = wire::Generate {
        prompt,
        outputs,
        seed,
        temperature: float("temperature", 0.)?,
        top_p: float("top_p", 0.95)?,
        min_p: float("min_p", 0.)?,
        top_k,
    };
    request.validate().map_err(|_| "native_options_invalid")?;
    Ok(Prepared {
        request: Some(request),
        chat,
        stream,
        keep_alive,
    })
}
pub fn scrub_json(value: &mut Value) {
    match value {
        Value::String(s) => s.zeroize(),
        Value::Array(a) => a.iter_mut().for_each(scrub_json),
        Value::Object(o) => o.values_mut().for_each(scrub_json),
        _ => {}
    }
}
pub struct Metrics {
    pub prompt: u32,
    pub outputs: u32,
    pub load_ns: u64,
    pub prompt_ns: u64,
    pub eval_ns: u64,
    pub total_ns: u64,
    pub eos: bool,
}
pub enum Event {
    Text(Zeroizing<Vec<u8>>),
    Done(Metrics),
    Failed(&'static str),
}
struct Worker {
    child: Child,
    input: Option<ChildStdin>,
    output: ChildStdout,
}
#[derive(Default)]
pub struct Runtime {
    worker: Mutex<Option<Worker>>,
    resident: AtomicBool,
}
impl Runtime {
    pub fn resident(&self) -> bool {
        self.resident.load(Ordering::Acquire)
    }
    pub async fn evict(&self) -> Result<()> {
        let mut state = self.worker.lock().await;
        if let Some(mut worker) = state.take() {
            worker.input.take();
            let result = tokio::time::timeout(Duration::from_secs(180), worker.child.wait()).await;
            if !matches!(result,Ok(Ok(status)) if status.success()) {
                let _ = worker.child.start_kill();
                tokio::time::timeout(Duration::from_secs(30), worker.child.wait())
                    .await
                    .map_err(|_| "native_worker_reap_failed")?
                    .map_err(|_| "native_worker_reap_failed")?;
                self.resident.store(false, Ordering::Release);
                return Err("native_worker_close_failed");
            }
        }
        self.resident.store(false, Ordering::Release);
        Ok(())
    }
    /// Called only while the global admission lease is held. A disconnected
    /// HTTP consumer is drained; model admission remains held until CLEAN/DONE.
    pub async fn run(
        &self,
        config: &Config,
        request: wire::Generate,
        events: mpsc::Sender<Event>,
    ) -> bool {
        let started = Instant::now();
        let mut state = self.worker.lock().await;
        let result = async {
            if state.is_none() {
                *state = Some(start(config).await?);
            }
            let worker = state.as_mut().ok_or("native_worker_unavailable")?;
            let input = worker.input.as_mut().ok_or("native_worker_unavailable")?;
            let controls = request.controls().map_err(|_| "native_request_invalid")?;
            input
                .write_all(wire::REQUEST_MAGIC)
                .await
                .map_err(|_| "native_worker_pipe_failed")?;
            input
                .write_all(&*controls)
                .await
                .map_err(|_| "native_worker_pipe_failed")?;
            input
                .write_all(&request.prompt)
                .await
                .map_err(|_| "native_worker_pipe_failed")?;
            input
                .flush()
                .await
                .map_err(|_| "native_worker_pipe_failed")?;
            drop(request);
            loop {
                let (kind, bytes) = read_response(&mut worker.output).await?;
                match kind {
                    wire::TEXT => {
                        self.resident.store(true, Ordering::Release);
                        let _ = events.send(Event::Text(bytes)).await;
                    }
                    wire::DONE if bytes.len() == 33 && bytes[32] <= 1 => {
                        let word = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                        let long = |at| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
                        self.resident.store(true, Ordering::Release);
                        let _ = events
                            .send(Event::Done(Metrics {
                                prompt: word(0),
                                outputs: word(4),
                                load_ns: long(8),
                                prompt_ns: long(16),
                                eval_ns: long(24),
                                eos: bytes[32] != 0,
                                total_ns: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                            }))
                            .await;
                        return Ok(());
                    }
                    wire::ERROR => return Err(worker_error(&bytes)),
                    _ => return Err("native_worker_protocol_failed"),
                }
            }
        }
        .await;
        if let Err(code) = result {
            let mut reaped = true;
            if let Some(mut worker) = state.take() {
                worker.input.take();
                let _ = worker.child.start_kill();
                reaped = matches!(
                    tokio::time::timeout(Duration::from_secs(30), worker.child.wait()).await,
                    Ok(Ok(_))
                );
            }
            self.resident.store(false, Ordering::Release);
            let _ = events.send(Event::Failed(code)).await;
            reaped
        } else {
            true
        }
    }
}
async fn start(config: &Config) -> Result<Worker> {
    crate::image_privacy::require_no_swap().map_err(|_| "native_worker_requires_no_swap")?;
    let parent = std::process::id();
    let mut command = Command::new(&config.executable);
    command
        .args([
            std::ffi::OsStr::new("--native-worker"),
            config.settings.as_os_str(),
        ])
        .current_dir("/")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", "/nonexistent")
        .env("MESA_SHADER_CACHE_DISABLE", "true")
        .env("__GL_SHADER_DISK_CACHE", "0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) != 0
                || libc::getppid() as u32 != parent
            {
                return Err(std::io::Error::other("native_parent_unavailable"));
            }
            Ok(())
        });
    }
    let mut child = command.spawn().map_err(|_| "native_worker_spawn_failed")?;
    let input = child.stdin.take();
    let output = child.stdout.take().ok_or("native_worker_pipe_failed")?;
    let mut worker = Worker {
        child,
        input,
        output,
    };
    match read_response(&mut worker.output).await {
        Ok((wire::READY, bytes)) if bytes.is_empty() => Ok(worker),
        result => {
            let code = match result {
                Ok((wire::ERROR, bytes)) => worker_error(&bytes),
                _ => "native_worker_start_failed",
            };
            let _ = worker.child.kill().await;
            let _ = worker.child.wait().await;
            Err(code)
        }
    }
}
fn worker_error(bytes: &[u8]) -> &'static str {
    match bytes {
        b"native_request_invalid" => "native_request_invalid",
        b"native_load_failed" => "native_load_failed",
        b"native_gpu_memory_busy" => "native_gpu_memory_busy",
        b"native_artifact_changed" => "native_artifact_changed",
        b"native_worker_requires_no_swap" => "native_worker_requires_no_swap",
        _ => "native_worker_failed",
    }
}
pub fn error_status(code: &str) -> axum::http::StatusCode {
    if code == "native_request_invalid" {
        axum::http::StatusCode::BAD_REQUEST
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    }
}
async fn read_response(output: &mut ChildStdout) -> Result<(u8, Zeroizing<Vec<u8>>)> {
    let mut header = [0; 9];
    output
        .read_exact(&mut header)
        .await
        .map_err(|_| "native_worker_pipe_failed")?;
    let length = u32::from_le_bytes(header[5..9].try_into().unwrap()) as usize;
    if &header[..4] != wire::RESPONSE_MAGIC || length > wire::MAX_FRAME_BYTES {
        return Err("native_worker_protocol_failed");
    }
    let mut bytes = Zeroizing::new(vec![0; length]);
    output
        .read_exact(&mut bytes)
        .await
        .map_err(|_| "native_worker_pipe_failed")?;
    Ok((header[4], bytes))
}
fn utc_now() -> String {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as libc::time_t;
    let mut tm = std::mem::MaybeUninit::<libc::tm>::uninit();
    // gmtime_r writes only caller-owned storage and has no shared mutable buffer.
    if unsafe { libc::gmtime_r(&timestamp, tm.as_mut_ptr()) }.is_null() {
        return "1970-01-01T00:00:00Z".into();
    }
    let tm = unsafe { tm.assume_init() };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    )
}
pub fn chunk(model: &str, chat: bool, text: &str, done: Option<&Metrics>) -> Value {
    let mut value = json!({"model":model,"created_at":utc_now(),"done":done.is_some()});
    if chat {
        value["message"] = json!({"role":"assistant","content":text});
    } else {
        value["response"] = text.into();
    }
    if let Some(m) = done {
        value["done_reason"] = if m.eos { "stop" } else { "length" }.into();
        value["load_duration"] = m.load_ns.into();
        value["prompt_eval_count"] = m.prompt.into();
        value["prompt_eval_duration"] = m.prompt_ns.into();
        value["eval_count"] = m.outputs.into();
        value["eval_duration"] = m.eval_ns.into();
        value["total_duration"] = m.total_ns.into();
    }
    value
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_preview_rejects_unimplemented_features_and_keeps_rendered_bytes() {
        let p = prepare(
            "/api/chat",
            &json!({"messages":[{"role":"user","content":"hello 🦦"}]}),
        )
        .unwrap();
        assert!(p.stream);
        assert_eq!(
            p.request.unwrap().prompt.as_slice(),
            "<|im_start|>user\nhello 🦦<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                .as_bytes()
        );
        for v in [
            json!({"prompt":"hi","format":"json"}),
            json!({"prompt":"hi","think":true}),
            json!({"prompt":"hi","options":{"num_ctx":4096}}),
            json!({"prompt":"hi","options":{"temperature":"x"}}),
            json!({"prompt":"hi","options":{"repeat_penalty":1.1}}),
        ] {
            assert!(prepare("/api/generate", &v).is_err());
        }
        assert!(
            prepare(
                "/api/chat",
                &json!({"messages":[{"role":"user","content":"hi","images":["x"]}]})
            )
            .is_err()
        );
        assert!(prepare("/v1/chat/completions", &json!({})).is_err());
    }
    #[test]
    fn raw_and_unload_preserve_explicit_behavior() {
        let p = prepare(
            "/api/generate",
            &json!({"prompt":"  raw\n","raw":true,"stream":false,"keep_alive":0}),
        )
        .unwrap();
        assert!(!p.stream && !p.keep_alive);
        assert_eq!(p.request.unwrap().prompt.as_slice(), b"  raw\n");
        assert!(
            prepare("/api/generate", &json!({"keep_alive":0}))
                .unwrap()
                .request
                .is_none()
        );
        assert!(
            prepare("/api/chat", &json!({"messages":[],"keep_alive":0}))
                .unwrap()
                .request
                .is_none()
        );
    }
}
