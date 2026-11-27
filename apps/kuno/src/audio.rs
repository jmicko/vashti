//! Speech recognition and synthesis (LFM3-Audio-2B, CPU) in a resident worker process.
//!
//! The worker is this executable (`kuno audio-worker <models>`), entered before any other
//! thread exists, with core dumps off, no swap, and no writable filesystem. It loads the
//! model once and answers requests on stdin/stdout; it writes nothing else, so content never
//! reaches a log. The gateway keeps it while it is used and stops it after five idle
//! minutes, so its RAM returns to language models. A client that goes away mid-request
//! stops the worker (the next request starts a fresh one).
//!
//! Frames: a request is `kind: u8`, then `len: u32` and that many bytes, twice (JSON
//! parameters, then the payload); a response is `status: u8` (0 ok, 1 error) and one
//! `len: u32`-prefixed body (the transcript, the WAV file, or the error code). The worker
//! announces itself with a response after loading.
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStdin, ChildStdout},
    sync::{Mutex, Semaphore},
};
use zeroize::Zeroize;

const TRANSCRIBE: u8 = 1;
const SPEAK: u8 = 2;
/// Audio a minute at a time; text a few sentences at a time.
const AUDIO_PIECE_SECONDS: usize = 60;
const TEXT_PIECE_CHARACTERS: usize = 400;
const IDLE: Duration = Duration::from_secs(300);
/// Requests waiting behind the one running.
const WAITING: usize = 4;
/// Worker error codes the gateway passes on; anything else becomes `audio_worker_failed`.
const CODES: [&str; 8] = [
    "audio_format_unsupported",
    "audio_too_short",
    "text_invalid",
    "model_file_invalid",
    "model_file_unavailable",
    "model_metadata_invalid",
    "model_tokenizer_invalid",
    "audio_request_invalid",
];

#[derive(Serialize, Deserialize)]
struct Params {
    /// Prompt names (`English`, `American English`), already validated by the gateway.
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    seed: u64,
}

// ---- Worker -----------------------------------------------------------------------------

/// `kuno audio-worker <models>`: serves requests until stdin closes.
pub fn run_worker() -> Result<(), &'static str> {
    crate::image_privacy::protect_process()?;
    crate::image_privacy::require_no_swap()?;
    crate::image_privacy::restrict_filesystem()?;
    let mut args = std::env::args().skip(2);
    let models = PathBuf::from(args.next().ok_or("missing_audio_models")?);
    let gpu = args.next();
    if args.next().is_some() {
        return Err("unknown_argument");
    }
    kuno_audio::use_physical_cores();
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    // A GPU that cannot be used (busy, too little memory) leaves the model on the CPU.
    let model = kuno_audio::Lfm3Audio::open_on(
        &models.join("LFM3-Audio-2B-Q8_0.gguf"),
        &models.join("mmproj-LFM3-Audio-2B-F16.gguf"),
        gpu.as_deref(),
    );
    let model = match model {
        Ok(model) => {
            // The ready message names the device (content-free), for discovery.
            respond(&mut output, Ok(model.device().as_bytes().to_vec()))?;
            model
        }
        Err(code) => {
            respond(&mut output, Err(code))?;
            return Err(code);
        }
    };
    loop {
        let mut kind = [0u8; 1];
        if input.read_exact(&mut kind).is_err() {
            return Ok(());
        }
        let params = read_part(&mut input)?;
        let mut payload = read_part(&mut input)?;
        let result = serde_json::from_slice::<Params>(&params)
            .map_err(|_| "audio_request_invalid")
            .and_then(|p| match kind[0] {
                TRANSCRIBE => transcribe(&model, &payload, &p).map(String::into_bytes),
                SPEAK => {
                    let text = std::str::from_utf8(&payload).map_err(|_| "text_invalid")?;
                    speak(&model, text, &p)
                }
                _ => Err("audio_request_invalid"),
            });
        payload.zeroize();
        respond(&mut output, result)?;
    }
}

fn read_part(input: &mut impl Read) -> Result<Vec<u8>, &'static str> {
    let mut len = [0u8; 4];
    input
        .read_exact(&mut len)
        .map_err(|_| "audio_request_invalid")?;
    let len = u32::from_le_bytes(len) as usize;
    if len > kuno_protocol::MAX_AUDIO_REQUEST_BYTES {
        return Err("audio_request_invalid");
    }
    let mut part = vec![0u8; len];
    input
        .read_exact(&mut part)
        .map_err(|_| "audio_request_invalid")?;
    Ok(part)
}

fn respond(
    output: &mut impl Write,
    result: Result<Vec<u8>, &'static str>,
) -> Result<(), &'static str> {
    let (status, mut body) = match result {
        Ok(body) => (0u8, body),
        Err(code) => (1u8, code.as_bytes().to_vec()),
    };
    let sent = output
        .write_all(&[status])
        .and_then(|_| output.write_all(&(body.len() as u32).to_le_bytes()))
        .and_then(|_| output.write_all(&body))
        .and_then(|_| output.flush());
    body.zeroize();
    sent.map_err(|_| "audio_worker_output_failed")
}

fn transcribe(
    model: &kuno_audio::Lfm3Audio,
    wav: &[u8],
    p: &Params,
) -> Result<String, &'static str> {
    let (samples, rate) = kuno_audio::wav::read(wav)?;
    let mut samples = kuno_audio::wav::resample(&samples, rate, 16_000);
    if samples.len() < 1_600 {
        return Err("audio_too_short");
    }
    let mut parts = Vec::new();
    for range in kuno_audio::split::audio(&samples, AUDIO_PIECE_SECONDS) {
        let seconds = range.len() / 16_000 + 1;
        let piece = &samples[range];
        if piece.len() < 1_600 {
            continue;
        }
        let text = model.transcribe(piece, p.language.as_deref(), 64 + seconds * 12)?;
        let text = text.trim();
        if !text.is_empty() {
            parts.push(text.to_owned());
        }
    }
    samples.zeroize();
    Ok(parts.join(" "))
}

fn speak(model: &kuno_audio::Lfm3Audio, text: &str, p: &Params) -> Result<Vec<u8>, &'static str> {
    let language = p.language.as_deref().unwrap_or("American English");
    let voice = p.voice.as_deref().unwrap_or("female");
    let mut audio = Vec::new();
    for (i, piece) in kuno_audio::split::text(text, TEXT_PIECE_CHARACTERS)
        .iter()
        .enumerate()
    {
        // About 15 characters a second at 12.5 frames a second, with ample room.
        let frames = 100 + piece.chars().count() * 3;
        let pcm = model.speak(
            piece,
            language,
            voice,
            p.seed.wrapping_add(i as u64),
            frames,
            |_| true,
        )?;
        audio.extend(pcm);
    }
    let wav = kuno_audio::wav::write(&audio, kuno_audio::SAMPLE_RATE);
    audio.zeroize();
    Ok(wav)
}

// ---- Gateway side -----------------------------------------------------------------------

struct Worker {
    _child: Child,
    input: ChildStdin,
    output: ChildStdout,
}

pub struct AudioRuntime {
    worker: Arc<Mutex<Option<Worker>>>,
    waiting: Arc<Semaphore>,
    uses: Arc<AtomicU64>,
    /// The device the last worker loaded onto ("cpu" or the GPU's name).
    device: Arc<std::sync::Mutex<Option<String>>>,
}

impl Default for AudioRuntime {
    fn default() -> Self {
        Self {
            worker: Arc::new(Mutex::new(None)),
            waiting: Arc::new(Semaphore::new(WAITING + 1)),
            uses: Arc::new(AtomicU64::new(0)),
            device: Arc::new(std::sync::Mutex::new(None)),
        }
    }
}

pub enum Job {
    Transcribe {
        wav: Vec<u8>,
        language: Option<String>,
    },
    Speak {
        text: String,
        language: Option<String>,
        voice: Option<String>,
        seed: u64,
    },
}

/// Whether `models` holds the verified collection (sizes; hashes are checked on fetch).
pub async fn ready(models: PathBuf) -> Result<(), &'static str> {
    tokio::task::spawn_blocking(move || {
        crate::collections::ready(&models, kuno_protocol::AUDIO_COLLECTION)
    })
    .await
    .map_err(|_| "audio_model_unavailable")?
    .map_err(|_| "audio_model_unavailable")
}

impl AudioRuntime {
    /// Runs `job` and returns the worker's body. `closed` resolves when the client is gone;
    /// the worker is then stopped rather than left computing for no one.
    pub async fn run(
        &self,
        executable: &Path,
        models: &Path,
        gpu: Option<&str>,
        job: Job,
        closed: impl std::future::Future<Output = ()>,
    ) -> Result<Vec<u8>, &'static str> {
        let _permit = self
            .waiting
            .clone()
            .try_acquire_owned()
            .map_err(|_| "audio_queue_full")?;
        let (kind, params, mut payload) = match job {
            Job::Transcribe { wav, language } => (
                TRANSCRIBE,
                Params {
                    language,
                    voice: None,
                    seed: 0,
                },
                wav,
            ),
            Job::Speak {
                text,
                language,
                voice,
                seed,
            } => (
                SPEAK,
                Params {
                    language,
                    voice,
                    seed,
                },
                text.into_bytes(),
            ),
        };
        let params = serde_json::to_vec(&params).map_err(|_| "audio_request_invalid")?;
        let mut slot = self.worker.lock().await;
        self.uses.fetch_add(1, Ordering::SeqCst);
        let result = tokio::select! {
            result = exchange(&mut slot, executable, models, gpu, &self.device, kind, &params, &payload) => result,
            _ = closed => Err("audio_cancelled"),
        };
        payload.zeroize();
        if result.is_err() {
            // A failed or abandoned exchange leaves the pipe mid-frame: start over next time.
            *slot = None;
        }
        drop(slot);
        self.evict_when_idle();
        result
    }

    /// The device the last worker loaded onto.
    pub fn device(&self) -> Option<String> {
        self.device.lock().ok().and_then(|d| d.clone())
    }

    fn evict_when_idle(&self) {
        let (worker, uses) = (self.worker.clone(), self.uses.clone());
        let seen = uses.load(Ordering::SeqCst);
        tokio::spawn(async move {
            tokio::time::sleep(IDLE).await;
            if uses.load(Ordering::SeqCst) == seen
                && let Ok(mut slot) = worker.try_lock()
                && uses.load(Ordering::SeqCst) == seen
            {
                *slot = None;
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
async fn exchange(
    slot: &mut Option<Worker>,
    executable: &Path,
    models: &Path,
    gpu: Option<&str>,
    device: &std::sync::Mutex<Option<String>>,
    kind: u8,
    params: &[u8],
    payload: &[u8],
) -> Result<Vec<u8>, &'static str> {
    if slot.is_none() {
        let (worker, name) = spawn(executable, models, gpu).await?;
        if let Ok(mut d) = device.lock() {
            *d = Some(name);
        }
        *slot = Some(worker);
    }
    let worker = slot.as_mut().unwrap();
    let io = async {
        let w = &mut worker.input;
        w.write_all(&[kind]).await?;
        w.write_all(&(params.len() as u32).to_le_bytes()).await?;
        w.write_all(params).await?;
        w.write_all(&(payload.len() as u32).to_le_bytes()).await?;
        w.write_all(payload).await?;
        w.flush().await?;
        Ok::<_, std::io::Error>(())
    };
    io.await.map_err(|_| "audio_worker_failed")?;
    read_response(&mut worker.output).await
}

async fn read_response(output: &mut ChildStdout) -> Result<Vec<u8>, &'static str> {
    let status = output.read_u8().await.map_err(|_| "audio_worker_failed")?;
    let len = output
        .read_u32_le()
        .await
        .map_err(|_| "audio_worker_failed")? as usize;
    if len > 512 * 1024 * 1024 {
        return Err("audio_worker_failed");
    }
    let mut body = vec![0u8; len];
    output
        .read_exact(&mut body)
        .await
        .map_err(|_| "audio_worker_failed")?;
    if status == 0 {
        return Ok(body);
    }
    let code = std::str::from_utf8(&body).unwrap_or("");
    Err(CODES
        .into_iter()
        .find(|c| *c == code)
        .unwrap_or("audio_worker_failed"))
}

async fn spawn(
    executable: &Path,
    models: &Path,
    gpu: Option<&str>,
) -> Result<(Worker, String), &'static str> {
    use std::process::Stdio;
    let mut command = tokio::process::Command::new(executable);
    command.arg("audio-worker").arg(models);
    if let Some(gpu) = gpu {
        command.arg(gpu);
    }
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", "/nonexistent")
        // No driver shader caches on disk (the sandbox has no writable path anyway).
        .env("MESA_SHADER_CACHE_DISABLE", "true")
        .env("__GL_SHADER_DISK_CACHE", "0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let parent = std::process::id();
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) != 0
                || libc::getppid() as u32 != parent
            {
                return Err(std::io::Error::other("audio_parent_unavailable"));
            }
            Ok(())
        });
    }
    let mut child = command.spawn().map_err(|_| "audio_worker_spawn_failed")?;
    let input = child.stdin.take().ok_or("audio_worker_spawn_failed")?;
    let mut output = child.stdout.take().ok_or("audio_worker_spawn_failed")?;
    // Loading reads 2.4 GB; a cold disk may take a while.
    match tokio::time::timeout(Duration::from_secs(300), read_response(&mut output)).await {
        Ok(Ok(name)) => Ok((
            Worker {
                _child: child,
                input,
                output,
            },
            String::from_utf8_lossy(&name).chars().take(128).collect(),
        )),
        Ok(Err(code)) => Err(code),
        Err(_) => Err("audio_worker_start_timeout"),
    }
}
