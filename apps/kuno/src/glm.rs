//! GLM 5.3 Flash on Kuno's own engine (crate `kuno-glm`): a native text model beside the
//! Ollama names, served by a resident worker that is this executable
//! (`kuno glm-worker <models> <pci> <context>`), entered before any other thread exists,
//! with core dumps off, no swap and no writable filesystem. It speaks the native text pipe
//! (`native_protocol`): the gateway renders the chat (`native_chat::render_glm`), the worker
//! tokenizes, decodes (speculatively, with the model's NextN block) and streams text; it
//! writes nothing else, so content never reaches a log.
//!
//! The model needs most of the B70 and ~90 GB of RAM (its experts), so the gateway clears
//! the other models before it runs and stops it when another model runs or after five idle
//! minutes. Between requests the worker keeps the sequence's state (in device memory, as
//! any decode does) and a saved point at the end of the last prompt: a conversation's next
//! request, which repeats that prompt, reads only what is new.
use std::{
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use zeroize::Zeroizing;

use crate::native_protocol as wire;

pub const IDLE: Duration = Duration::from_secs(300);
/// Tokens drafted per step by the NextN block.
const DRAFTS: usize = 2;
/// Positions beyond the context for a step's drafts.
const SLACK: usize = 8;

fn default_context() -> u32 {
    wire::MAX_CONTEXT
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The name clients use (`kuno-` prefixed, like the other native model).
    pub model: String,
    /// The directory of the GGUF shards (as the service sees it).
    pub models: PathBuf,
    pub pci_address: String,
    /// Positions the worker holds (requests may ask for fewer with num_ctx).
    #[serde(default = "default_context")]
    pub context: u32,
}

impl Config {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.model.starts_with("kuno-")
            || self.model.len() > 128
            || !self
                .model
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
            || !self.models.is_absolute()
            || self.pci_address.len() != 12
            || !(2048..=wire::MAX_CONTEXT).contains(&self.context)
        {
            return Err("glm_configuration_invalid");
        }
        Ok(())
    }

    pub fn launch(&self) -> Result<crate::native_text::Launch, &'static str> {
        Ok(crate::native_text::Launch {
            executable: std::env::current_exe().map_err(|_| "executable_unavailable")?,
            args: vec![
                "glm-worker".into(),
                self.models.clone().into(),
                self.pci_address.clone().into(),
                self.context.to_string().into(),
            ],
        })
    }

    /// The shards' total size and a digest of the pinned shard list (or of their names and
    /// sizes), for the catalog.
    fn artifact(&self) -> (u64, String) {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        let mut size = 0u64;
        let mut names: Vec<(String, u64)> = std::fs::read_dir(&self.models)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let len = e.metadata().ok()?.len();
                name.ends_with(".gguf").then_some((name, len))
            })
            .collect();
        names.sort();
        match std::fs::read(self.models.join("PINS.txt")) {
            Ok(pins) => hash.update(&pins),
            Err(_) => {
                for (name, len) in &names {
                    hash.update(name.as_bytes());
                    hash.update(len.to_le_bytes());
                }
            }
        }
        for (_, len) in &names {
            size += len;
        }
        let digest = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        (size, digest)
    }

    pub fn metadata(&self) -> Value {
        let (size, digest) = self.artifact();
        json!({"name":self.model,"model":self.model,"size":size,"digest":digest,
            "modified_at":"2026-11-24T00:00:00Z",
            "details":{"format":"gguf","family":"glm5next","families":["glm5next"],
                "parameter_size":"320B","quantization_level":"IQ3_XXS"},
            "kuno_backend":"rust-vulkan"})
    }

    pub fn show(&self) -> Value {
        json!({"license":"See model artifact license",
            "parameters":format!("temperature 1\ntop_p 0.95\ntop_k 40\nnum_ctx {}\nnum_predict 1024", self.context),
            "template":"{{ .Prompt }}","capabilities":["completion","thinking","tools"],
            "details":self.metadata()["details"],
            "model_info":{"general.architecture":"glm5next","glm5next.context_length":self.context},
            "kuno_backend":"rust-vulkan"})
    }
}

// ---- Worker -----------------------------------------------------------------------------

fn respond(output: &mut impl Write, kind: u8, bytes: &[u8]) -> Result<(), &'static str> {
    wire::write_response(output, kind, bytes).map_err(|_| "glm_worker_output_failed")
}

/// `kuno glm-worker <models> <pci> <context>`: serves requests until stdin closes.
pub fn run_worker() -> Result<(), &'static str> {
    crate::image_privacy::protect_process()?;
    crate::image_privacy::require_no_swap()?;
    crate::image_privacy::restrict_filesystem()?;
    let mut args = std::env::args().skip(2);
    let models = PathBuf::from(args.next().ok_or("missing_glm_models")?);
    let pci = args.next().ok_or("missing_glm_device")?;
    let context: usize = args
        .next()
        .and_then(|c| c.parse().ok())
        .ok_or("missing_glm_context")?;
    if args.next().is_some() {
        return Err("unknown_argument");
    }
    kuno_audio::use_physical_cores();
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&models)
        .map_err(|_| "native_load_failed")?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    shards.sort();
    let loaded = (|| {
        let model = kuno_glm::Model::open(&shards).map_err(|_| "native_load_failed")?;
        let tokenizer = model.tokenizer().map_err(|_| "native_load_failed")?;
        Ok::<_, &'static str>((model, tokenizer))
    })();
    let (model, tokenizer) = match loaded {
        Ok(v) => v,
        Err(code) => {
            respond(&mut output, wire::ERROR, code.as_bytes())?;
            return Err(code);
        }
    };
    let select = || kuno_image::engine::gpu::Select::Pci(pci.clone());
    let gpu = kuno_image::engine::gpu::Gpu::open_side(select())
        .or_else(|_| kuno_image::engine::gpu::Gpu::open(select()));
    let engine = gpu.map_err(|_| "native_gpu_memory_busy").and_then(|gpu| {
        kuno_glm::engine::Engine::new(&model, gpu, None, context + SLACK)
            .map_err(|_| "native_load_failed")
    });
    let mut engine = match engine {
        Ok(e) => e,
        Err(code) => {
            respond(&mut output, wire::ERROR, code.as_bytes())?;
            return Err(code);
        }
    };
    // The tokens that end a reply: end of text, the next user turn, a tool result.
    let mut stops = vec![tokenizer.eos_token_id()];
    for marker in ["<|user|>", "<|observation|>"] {
        if let Ok(ids) = tokenizer.encode(marker, true)
            && ids.len() == 1
        {
            stops.push(ids[0]);
        }
    }
    respond(&mut output, wire::READY, &[])?;
    let mut saved: Option<Zeroizing<Vec<u32>>> = None;
    loop {
        let request = match wire::read_request(&mut input) {
            Ok(Some(r)) => r,
            Ok(None) => return Ok(()),
            Err(_) => return Err("native_request_invalid"),
        };
        serve(
            &mut engine,
            &tokenizer,
            &stops,
            context,
            request,
            &mut saved,
            &mut output,
        )?;
    }
}

/// One request: text frames as tokens are decided, then DONE (or REJECTED before running).
/// An engine failure ends the worker (its state is then unknown).
fn serve(
    engine: &mut kuno_glm::engine::Engine,
    tokenizer: &kuno_text::Qwen35Tokenizer,
    stops: &[u32],
    context: usize,
    request: wire::Generate,
    saved: &mut Option<Zeroizing<Vec<u32>>>,
    output: &mut impl Write,
) -> Result<(), &'static str> {
    let started = Instant::now();
    let done = |output: &mut dyn Write, m: [u64; 6]| {
        let mut b = Vec::with_capacity(33);
        b.extend_from_slice(&(m[0] as u32).to_le_bytes());
        b.extend_from_slice(&(m[1] as u32).to_le_bytes());
        for v in &m[2..5] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.push(m[5] as u8);
        let mut output = output;
        wire::write_response(&mut output, wire::DONE, &b).map_err(|_| "glm_worker_output_failed")
    };
    let failed = |output: &mut dyn Write, code: &'static str| -> Result<(), &'static str> {
        let mut output = output;
        let _ = wire::write_response(&mut output, wire::ERROR, code.as_bytes());
        Err(code)
    };
    if request.prompt.is_empty() {
        // A preload: every block's experts into memory now.
        if engine.load_all().is_err() {
            return failed(output, "native_load_failed");
        }
        let ns = started.elapsed().as_nanos() as u64;
        return done(output, [0, 0, ns, 0, 0, 0]);
    }
    let Ok(text) = std::str::from_utf8(&request.prompt) else {
        return respond(output, wire::REJECTED, b"native_request_invalid");
    };
    let tokens = match tokenizer.encode(text, true) {
        Ok(t) => Zeroizing::new(t),
        Err(_) => return respond(output, wire::REJECTED, b"native_request_invalid"),
    };
    let limit = (request.context as usize).min(context);
    if tokens.is_empty() || tokens.len() + request.outputs as usize > limit {
        return respond(output, wire::REJECTED, b"native_context_exceeded");
    }
    // A conversation's next request repeats the last prompt: read only what follows it.
    let reuse = saved
        .as_ref()
        .is_some_and(|p| p.len() < tokens.len() && tokens.starts_with(p));
    let from = if reuse {
        match engine.rewind() {
            Ok(true) => saved.as_ref().map_or(0, |p| p.len()),
            Ok(false) => 0,
            Err(_) => return failed(output, "native_worker_failed"),
        }
    } else {
        0
    };
    *saved = None;
    if from == 0 {
        engine.forget_point();
        if engine.reset().is_err() {
            return failed(output, "native_worker_failed");
        }
    }
    let feed = &tokens[from..];
    // A short prompt runs a token batch at a time, which needs every block loaded first.
    if feed.len() < 32 && engine.load_all().is_err() {
        return failed(output, "native_load_failed");
    }
    let load_ns = started.elapsed().as_nanos() as u64;
    let mut sampler = kuno_glm::engine::Sampler::new(
        request.seed,
        request.temperature,
        request.top_k.max(0) as usize,
        request.top_p,
        request.min_p,
    );
    let mut decoder = tokenizer.stream_decoder();
    let mut first: Option<Instant> = None;
    let mut outputs = 0u64;
    let mut eos = false;
    let mut broken: Option<&'static str> = None;
    let prefill = Instant::now();
    let result = engine.decode(
        feed,
        request.outputs as usize,
        DRAFTS,
        Some(&mut sampler),
        true,
        &mut |token| {
            first.get_or_insert_with(Instant::now);
            if stops.contains(&token) {
                eos = true;
                return false;
            }
            outputs += 1;
            let piece = match tokenizer
                .token_bytes(token)
                .and_then(|b| decoder.push_bytes(b))
            {
                Ok(p) => Zeroizing::new(p),
                Err(_) => {
                    broken = Some("native_worker_failed");
                    return false;
                }
            };
            for part in piece.as_bytes().chunks(wire::MAX_FRAME_BYTES) {
                if wire::write_response(output, wire::TEXT, part).is_err() {
                    broken = Some("glm_worker_output_failed");
                    return false;
                }
            }
            true
        },
    );
    if let Some(code) = broken {
        return failed(output, code);
    }
    if result.is_err() {
        return failed(output, "native_worker_failed");
    }
    // The saved point is the end of this prompt.
    *saved = Some(tokens);
    let rest = Zeroizing::new(decoder.finish());
    if !rest.is_empty() {
        respond(output, wire::TEXT, rest.as_bytes())?;
    }
    let first = first.unwrap_or_else(Instant::now);
    let prompt_ns = first.duration_since(prefill).as_nanos() as u64;
    let eval_ns = first.elapsed().as_nanos() as u64;
    let prompt = saved.as_ref().map_or(0, |t| t.len()) as u64;
    done(
        output,
        [prompt, outputs, load_ns, prompt_ns, eval_ns, eos as u64],
    )
}
