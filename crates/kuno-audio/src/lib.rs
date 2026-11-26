//! LFM3-Audio-2B on the CPU for Kuno: speech recognition and speech synthesis, following
//! llama.cpp's implementation (`src/models/lfm3.cpp`, `tools/mtmd/models/lfm3a*.cpp`,
//! `mtmd-helper-gen.cpp`) step for step.
//!
//! - the audio frontend: 16 kHz samples to 128 log-mel rows (400-sample periodic Hann window,
//!   hop 160, Slaney mel, `(log10 + 4) / 4`);
//! - the encoder: three stride-2 convolutions (8x in time), 17 LFM blocks (gated short
//!   convolutions and sliding-window attention over 32 positions), an MLP projector to the
//!   backbone's width; one row per 80 ms;
//! - the backbone: 24 LFM blocks (short convolutions, full and 256-window GQA attention with
//!   per-head gates and sandwich norms), the text head tied to the embeddings;
//! - the depth decoder: from a backbone state, 12 speech codes per 80 ms frame, revealed four
//!   at a time over three bidirectional passes (MaskGIT); a frame feeds back as the sum of its
//!   code embeddings;
//! - the codec: the codes to STFT frames (8 LFM blocks over six positions a frame), then an
//!   inverse STFT to 24 kHz audio.
use std::path::Path;

use kuno_text::{
    GgufFile, GgufLimits,
    tokenizer::{Qwen35Tokenizer, TokenizerLimits},
};

pub mod block;
pub mod split;
pub mod tensor;
pub mod wav;

use block::{Attention, Block, Conv, Op, State};
use tensor::{Linear, gelu_erf, layer_norm, rms_norm, silu, vector};

/// Sizes the global thread pool to one thread per physical core (once, before any work):
/// decoding is bound by memory bandwidth and many small synchronized steps, so a second
/// thread per core only adds contention (speech 0.86x to 0.97x real time on a 16-core
/// Ryzen; recognition unchanged).
pub fn use_physical_cores() {
    let cores = physical_cores();
    if cores > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(cores)
            .build_global();
    }
}

/// Distinct (package, core) pairs among the online CPUs; 0 if the topology is unreadable.
fn physical_cores() -> usize {
    let mut seen = std::collections::BTreeSet::new();
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") else {
        return 0;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(id) = name.to_str().and_then(|n| n.strip_prefix("cpu")) else {
            continue;
        };
        if id.is_empty() || !id.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let topology = entry.path().join("topology");
        let read = |f: &str| std::fs::read_to_string(topology.join(f)).ok();
        if let (Some(package), Some(core)) = (read("physical_package_id"), read("core_id")) {
            seen.insert((package.trim().to_owned(), core.trim().to_owned()));
        }
    }
    seen.len()
}

pub type Result<T> = std::result::Result<T, &'static str>;

const EPS: f32 = 1e-5;
/// Speech codebooks per frame and codes per codebook (the last ends the audio).
pub const CODEBOOKS: usize = 12;
const CODES: usize = 2017;
/// Codebooks revealed by each depth-decoder pass.
const SCHEDULE: [usize; 3] = [4, 4, 4];
/// Output audio.
pub const SAMPLE_RATE: u32 = 24_000;
const N_FFT: usize = 1280;
const HOP: usize = 320;
const FRAME_STACK: usize = 6;

fn open(path: &Path) -> Result<GgufFile> {
    let limits = GgufLimits {
        max_tensor_read_bytes: 1 << 32,
        ..GgufLimits::default()
    };
    GgufFile::open(path, limits).map_err(|_| "model_file_invalid")
}

/// Loads an LFM block. `p` is the tensor prefix (e.g. `a.blk.3.`); conv layers have
/// `conv_dw`/`conv_pw1`/`conv_pw2` (or the backbone's `shortconv.*`), attention layers q/k/v.
struct Names {
    ln1: &'static str,
    ln2: &'static str,
    post_op: &'static str,
    post_ffn: &'static str,
    q: &'static str,
    k: &'static str,
    v: &'static str,
    o: &'static str,
    q_norm: &'static str,
    k_norm: &'static str,
    gate: &'static str,
    conv_in: &'static str,
    conv_taps: &'static str,
    conv_out: &'static str,
}
const MTMD: Names = Names {
    ln1: "ln1.weight",
    ln2: "ln2.weight",
    post_op: "attn_post_norm.weight",
    post_ffn: "ffn_post_norm.weight",
    q: "attn_q.weight",
    k: "attn_k.weight",
    v: "attn_v.weight",
    o: "attn_out.weight",
    q_norm: "attn_q_norm.weight",
    k_norm: "attn_k_norm.weight",
    gate: "attn_gate.weight",
    conv_in: "conv_pw1.weight",
    conv_taps: "conv_dw.weight",
    conv_out: "conv_pw2.weight",
};
const LLM: Names = Names {
    ln1: "attn_norm.weight",
    ln2: "ffn_norm.weight",
    post_op: "post_attention_norm.weight",
    post_ffn: "post_ffw_norm.weight",
    q: "attn_q.weight",
    k: "attn_k.weight",
    v: "attn_v.weight",
    o: "attn_output.weight",
    q_norm: "attn_q_norm.weight",
    k_norm: "attn_k_norm.weight",
    gate: "attn_gate.weight",
    conv_in: "shortconv.in_proj.weight",
    conv_taps: "shortconv.conv.weight",
    conv_out: "shortconv.out_proj.weight",
};

struct AttnParams {
    rope_base: f32,
    factors: Option<Vec<f32>>,
    window: Option<usize>,
}

fn load_block(
    g: &GgufFile,
    p: &str,
    n: &Names,
    width: usize,
    attn: impl Fn(bool) -> AttnParams,
) -> Result<Block> {
    let t = |s: &str| format!("{p}{s}");
    let has = |s: &str| g.tensor(&t(s)).is_some();
    let op = if has(n.q) {
        let q = Linear::load(g, &t(n.q))?;
        let k = Linear::load(g, &t(n.k))?;
        let q_norm = vector(g, &t(n.q_norm))?;
        let head_dim = q_norm.len();
        let heads = q.rows() / head_dim;
        let kv_heads = k.rows() / head_dim;
        let ap = attn(true);
        Op::Attention(Attention {
            q,
            k,
            v: Linear::load(g, &t(n.v))?,
            o: Linear::load(g, &t(n.o))?,
            q_norm,
            k_norm: vector(g, &t(n.k_norm))?,
            gate: if has(n.gate) {
                Some(Linear::load(g, &t(n.gate))?)
            } else {
                None
            },
            heads,
            kv_heads,
            head_dim,
            rope_base: ap.rope_base,
            factors: ap.factors,
            window: ap.window,
        })
    } else {
        let taps = vector(g, &t(n.conv_taps))?;
        Op::Conv(Conv {
            in_proj: Linear::load(g, &t(n.conv_in))?,
            kernel: taps.len() / width,
            taps,
            out_proj: Linear::load(g, &t(n.conv_out))?,
        })
    };
    Ok(Block {
        width,
        eps: EPS,
        ln1: vector(g, &t(n.ln1))?,
        ln2: vector(g, &t(n.ln2))?,
        post_op: if has(n.post_op) {
            Some(vector(g, &t(n.post_op))?)
        } else {
            None
        },
        post_ffn: if has(n.post_ffn) {
            Some(vector(g, &t(n.post_ffn))?)
        } else {
            None
        },
        op,
        gate: Linear::load(g, &t("ffn_gate.weight"))?,
        up: Linear::load(g, &t("ffn_up.weight"))?,
        down: Linear::load(g, &t("ffn_down.weight"))?,
    })
}

/// A 3x3 stride-2 convolution's weights: `[out][in][freq][time]`.
struct Conv2d {
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    depthwise: bool,
}

pub struct Lfm3Audio {
    tokenizer: Qwen35Tokenizer,
    embed: Linear,
    backbone: Vec<Block>,
    out_norm: Vec<f32>,
    // Encoder.
    down: [Conv2d; 3],
    point: [Conv2d; 2],
    pre_out: (Linear, Vec<f32>),
    encoder: Vec<Block>,
    post_ln: Vec<f32>,
    proj: (Vec<f32>, Vec<f32>, Linear, Vec<f32>, Linear, Vec<f32>),
    mel_filters: Vec<f32>,
    // Depth decoder.
    depth: Vec<Block>,
    proj_in: (Linear, Vec<f32>),
    code_embd: Linear,
    mask_embd: Vec<f32>,
    depth_norm: Vec<f32>,
    head_norm: Vec<f32>,
    feedback: Linear,
    // Codec.
    fsq: Vec<f32>,
    fsq_out: (Linear, Vec<f32>),
    codec_in: (Linear, Vec<f32>),
    codec: Vec<Block>,
    codec_norm: Vec<f32>,
    codec_out: (Linear, Vec<f32>),
    tokens: Tokens,
}

/// Control tokens of the prompts and the generation loop.
struct Tokens {
    bos: u32,
    im_end: u32,
    audio_start: u32,
}

impl Lfm3Audio {
    /// The backbone GGUF and the `mmproj` GGUF (encoder, depth decoder, codec).
    pub fn open(backbone: &Path, mmproj: &Path) -> Result<Self> {
        let g = open(backbone)?;
        let a = open(mmproj)?;
        let arch = g
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.as_str());
        if arch != Some("lfm3") {
            return Err("model_file_invalid");
        }
        let limits = TokenizerLimits {
            max_special_tokens: 4096,
            ..TokenizerLimits::default()
        };
        let tokenizer =
            Qwen35Tokenizer::from_gguf(&g, limits).map_err(|_| "model_tokenizer_invalid")?;
        let meta = g.metadata();
        let swa_pattern: Vec<bool> = meta
            .get("lfm3.attention.sliding_window_pattern")
            .and_then(|v| v.as_array())
            .ok_or("model_metadata_invalid")?
            .iter()
            .map(|v| matches!(v, kuno_text::MetadataValue::Bool(true)))
            .collect();
        let factors = vector(&g, "rope_freqs.weight")?;
        let layers = swa_pattern.len();
        let backbone = (0..layers)
            .map(|il| {
                let swa = swa_pattern[il];
                load_block(&g, &format!("blk.{il}."), &LLM, 2048, |_| AttnParams {
                    rope_base: if swa { 10_000.0 } else { 1_000_000.0 },
                    factors: (!swa).then(|| factors.clone()),
                    window: swa.then_some(256),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let conv2d = |i: usize, depthwise: bool| -> Result<Conv2d> {
            let w = vector(&a, &format!("a.conv1d.{i}.weight"))?;
            let b = vector(&a, &format!("a.conv1d.{i}.bias"))?;
            let cout = b.len();
            let cin = if depthwise || i == 0 {
                1
            } else {
                w.len() / cout
            };
            Ok(Conv2d {
                w,
                b,
                cin,
                cout,
                depthwise,
            })
        };
        let lin = |name: &str| Linear::load(&a, name);
        let vecs = |name: &str| vector(&a, name);
        let enc_blocks = (0..17)
            .map(|il| {
                load_block(&a, &format!("a.blk.{il}."), &MTMD, 768, |_| AttnParams {
                    rope_base: 1_000_000.0,
                    factors: None,
                    window: Some(32),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let depth = (0..3)
            .map(|il| {
                load_block(&a, &format!("a.gen.code.blk.{il}."), &MTMD, 1024, |_| {
                    AttnParams {
                        rope_base: 1_000_000.0,
                        factors: None,
                        window: None,
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let codec = (0..8)
            .map(|il| {
                load_block(&a, &format!("a.gen.wav.tfm.blk.{il}."), &MTMD, 512, |_| {
                    AttnParams {
                        rope_base: 1_000_000.0,
                        factors: None,
                        window: Some(30),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let special = |text: &str| -> Result<u32> {
            let ids = tokenizer
                .encode(text, true)
                .map_err(|_| "model_tokenizer_invalid")?;
            match ids.as_slice() {
                [id] => Ok(*id),
                _ => Err("model_tokenizer_invalid"),
            }
        };
        let tokens = Tokens {
            bos: tokenizer.bos_token_id(),
            im_end: special("<|im_end|>")?,
            audio_start: special("<|audio_start|>")?,
        };
        Ok(Self {
            embed: Linear::load(&g, "token_embd.weight")?,
            backbone,
            out_norm: vector(&g, "output_norm.weight")?,
            down: [conv2d(0, false)?, conv2d(2, true)?, conv2d(5, true)?],
            point: [conv2d(3, false)?, conv2d(6, false)?],
            pre_out: (
                lin("a.pre_encode.out.weight")?,
                vecs("a.pre_encode.out.bias")?,
            ),
            encoder: enc_blocks,
            post_ln: vecs("a.post_ln.weight")?,
            proj: (
                vecs("mm.a.mlp.0.weight")?,
                vecs("mm.a.mlp.0.bias")?,
                lin("mm.a.mlp.1.weight")?,
                vecs("mm.a.mlp.1.bias")?,
                lin("mm.a.mlp.3.weight")?,
                vecs("mm.a.mlp.3.bias")?,
            ),
            mel_filters: mel_filterbank(128, 400, 16_000),
            depth,
            proj_in: (
                lin("a.gen.code.proj_in.weight")?,
                vecs("a.gen.code.proj_in.bias")?,
            ),
            code_embd: lin("a.gen.code.embd.weight")?,
            mask_embd: vecs("a.gen.code.mask_embd")?,
            depth_norm: vecs("a.gen.code.output_norm.weight")?,
            head_norm: vecs("a.gen.code.head_norm.weight")?,
            feedback: lin("a.gen.code.out_embd.weight")?,
            fsq: vecs("a.gen.wav.quant.first.codebook.weight")?,
            fsq_out: (
                lin("a.gen.wav.quant.first.out_proj.weight")?,
                vecs("a.gen.wav.quant.first.out_proj.bias")?,
            ),
            codec_in: (
                lin("a.gen.wav.tfm.in_proj.weight")?,
                vecs("a.gen.wav.tfm.in_proj.bias")?,
            ),
            codec,
            codec_norm: vecs("a.gen.wav.tfm.output_norm.weight")?,
            codec_out: (
                lin("a.gen.wav.tfm.out_proj.weight")?,
                vecs("a.gen.wav.tfm.out_proj.bias")?,
            ),
            tokenizer,
            tokens,
        })
    }

    // ---- Encoder ------------------------------------------------------------------------

    /// 16 kHz mono samples to backbone-width rows (one per 80 ms).
    pub fn encode_audio(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let (mel, frames) = log_mel(samples, &self.mel_filters);
        // [channel][time][freq], starting from one channel.
        let (mut x, mut t, mut f) = (mel_to_tf(&mel, frames), frames, 128usize);
        let mut ch = 1;
        for (i, conv) in self.down.iter().enumerate() {
            let (y, t2, f2) = conv_down(&x, ch, t, f, conv);
            (x, t, f) = (y, t2, f2);
            ch = conv.cout;
            if i > 0 {
                x = pointwise(&x, ch, t * f, &self.point[i - 1]);
            }
            x.iter_mut().for_each(|v| *v = silu(*v));
        }
        // Each position's features: freq fastest, then channel.
        let mut rows = vec![0f32; t * f * ch];
        for c in 0..ch {
            for ti in 0..t {
                for fi in 0..f {
                    rows[ti * f * ch + c * f + fi] = x[(c * t + ti) * f + fi];
                }
            }
        }
        let mut h = self.pre_out.0.forward_bias(&rows, &self.pre_out.1);
        let mut states = vec![State::default(); self.encoder.len()];
        for (b, s) in self.encoder.iter().zip(&mut states) {
            b.forward(&mut h, s, true);
        }
        rms_norm(&mut h, Some(&self.post_ln), 768, EPS);
        let (w0, b0, l1, b1, l3, b3) = &self.proj;
        layer_norm(&mut h, w0, b0, 1e-5);
        let mut m = l1.forward_bias(&h, b1);
        m.iter_mut().for_each(|v| *v = gelu_erf(*v));
        Ok(l3.forward_bias(&m, b3))
    }

    // ---- Backbone -----------------------------------------------------------------------

    fn embed_tokens(&self, ids: &[u32]) -> Vec<f32> {
        ids.iter()
            .flat_map(|id| self.embed.row(*id as usize))
            .collect()
    }
    /// Runs rows through the backbone; returns the last row's normed state.
    fn backbone(&self, rows: &mut [f32], states: &mut [State]) -> Vec<f32> {
        for (b, s) in self.backbone.iter().zip(states.iter_mut()) {
            b.forward(rows, s, true);
        }
        let mut last = rows[rows.len() - 2048..].to_vec();
        rms_norm(&mut last, Some(&self.out_norm), 2048, EPS);
        last
    }
    fn logits(&self, state: &[f32]) -> Vec<f32> {
        self.embed.forward(state)
    }

    // ---- Recognition --------------------------------------------------------------------

    /// The transcript of 16 kHz mono `samples`; `language` names it in the system prompt
    /// ("English"), or the model infers it.
    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        max_tokens: usize,
    ) -> Result<String> {
        let audio = self.encode_audio(samples)?;
        let system = match language {
            Some(l) => format!("Perform ASR in {l}."),
            None => "Perform ASR.".into(),
        };
        let before = format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n");
        let after = "<|im_end|>\n<|im_start|>assistant\n";
        let enc = |s: &str| self.tokenizer.encode(s, true).map_err(|_| "text_invalid");
        let mut ids = vec![self.tokens.bos];
        ids.extend(enc(&before)?);
        let mut rows = self.embed_tokens(&ids);
        rows.extend(&audio);
        rows.extend(self.embed_tokens(&enc(after)?));
        let mut states = vec![State::default(); self.backbone.len()];
        let mut state = self.backbone(&mut rows, &mut states);
        let mut out = Vec::new();
        for _ in 0..max_tokens {
            let logits = self.logits(&state);
            let next = argmax(&logits) as u32;
            if next == self.tokens.im_end || next == self.tokenizer.eos_token_id() {
                break;
            }
            out.push(next);
            let mut row = self.embed_tokens(&[next]);
            state = self.backbone(&mut row, &mut states);
        }
        self.tokenizer.decode(&out).map_err(|_| "text_invalid")
    }

    // ---- Synthesis ----------------------------------------------------------------------

    /// 24 kHz mono speech of `text` in `language` (the template's names, e.g. "American
    /// English") with a `female` or `male` voice. `on_audio` receives samples as frames are
    /// decoded and may stop the generation by returning false.
    pub fn speak(
        &self,
        text: &str,
        language: &str,
        voice: &str,
        seed: u64,
        max_frames: usize,
        mut on_audio: impl FnMut(&[f32]) -> bool,
    ) -> Result<Vec<f32>> {
        let prompt = format!(
            "<|im_start|>system\nPerform TTS in {language} using a {voice} voice.<|im_end|>\n\
             <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|audio_start|>"
        );
        let mut ids = vec![self.tokens.bos];
        ids.extend(
            self.tokenizer
                .encode(&prompt, true)
                .map_err(|_| "text_invalid")?,
        );
        debug_assert_eq!(*ids.last().unwrap(), self.tokens.audio_start);
        let mut states = vec![State::default(); self.backbone.len()];
        let mut rows = self.embed_tokens(&ids);
        let mut state = self.backbone(&mut rows, &mut states);
        let mut rng = Rng::new(seed);
        let mut history: Vec<[usize; CODEBOOKS]> = Vec::new();
        let mut codec = CodecState::new(self.codec.len());
        let mut audio = Vec::new();
        for _ in 0..max_frames {
            let codes = self.sample_frame(&state, &history, &mut rng);
            if codes.contains(&(CODES - 1)) {
                break;
            }
            history.push(codes);
            let mut fb = vec![0f32; 2048];
            for (cb, &code) in codes.iter().enumerate() {
                for (f, v) in fb.iter_mut().zip(self.feedback.row(cb * CODES + code)) {
                    *f += v;
                }
            }
            // The frame's audio and the next backbone step are independent: run them together.
            let (pcm, next) = rayon::join(
                || self.decode_frame(&codes, &mut codec),
                || self.backbone(&mut fb, &mut states),
            );
            state = next;
            audio.extend(&pcm);
            if !on_audio(&pcm) {
                return Ok(audio);
            }
        }
        let tail = codec.istft.flush();
        on_audio(&tail);
        audio.extend(tail);
        Ok(audio)
    }

    /// One frame's codes from a backbone state: three depth-decoder passes, each revealing
    /// four codebooks in a shuffled order, sampled with top-k 128 at temperature 1 and a
    /// repetition penalty of 1.1 over each codebook's earlier codes.
    fn sample_frame(
        &self,
        h: &[f32],
        history: &[[usize; CODEBOOKS]],
        rng: &mut Rng,
    ) -> [usize; CODEBOOKS] {
        let mut order: Vec<usize> = (0..CODEBOOKS).collect();
        rng.shuffle(&mut order);
        let mut codes: [Option<usize>; CODEBOOKS] = [None; CODEBOOKS];
        let base = self.proj_in.0.forward_bias(h, &self.proj_in.1);
        let mut k = 0;
        for &pick in &SCHEDULE {
            let chosen = order[k..k + pick].to_vec();
            let logits = self.depth_logits(&base, &codes, &chosen);
            for (n, &cb) in chosen.iter().enumerate() {
                k += 1;
                let mut l = logits[n * CODES..][..CODES].to_vec();
                let mut seen = vec![false; CODES];
                for frame in history.iter().rev().take(4096) {
                    let c = frame[cb];
                    if !seen[c] {
                        seen[c] = true;
                        l[c] = if l[c] > 0.0 { l[c] / 1.1 } else { l[c] * 1.1 };
                    }
                }
                codes[cb] = Some(rng.sample_top_k(&l, 128, 1.0));
            }
        }
        codes.map(|c| c.unwrap_or(0))
    }

    /// The logits of codebooks `wanted` (in that order) after one depth-decoder pass.
    fn depth_logits(
        &self,
        base: &[f32],
        codes: &[Option<usize>; CODEBOOKS],
        wanted: &[usize],
    ) -> Vec<f32> {
        let w = 1024;
        let mut x = base.to_vec();
        for (cb, code) in codes.iter().enumerate() {
            let add = match code {
                Some(c) => self.code_embd.row(cb * CODES + c),
                None => self.mask_embd.clone(),
            };
            for (v, a) in x[cb * w..][..w].iter_mut().zip(add) {
                *v += a;
            }
        }
        for b in &self.depth {
            b.forward(&mut x, &mut State::default(), false);
        }
        rms_norm(&mut x, Some(&self.depth_norm), w, EPS);
        for (cb, row) in x.chunks_exact_mut(w).enumerate() {
            rms_norm(row, Some(&self.head_norm[cb * w..][..w]), w, EPS);
        }
        // Each codebook's logits from its own block of the tied embedding table.
        use rayon::prelude::*;
        (0..wanted.len() * CODES)
            .into_par_iter()
            .map(|i| {
                let cb = wanted[i / CODES];
                tensor::dot_row(&self.code_embd, cb * CODES + i % CODES, &x[cb * w..][..w])
            })
            .collect()
    }

    /// One frame's codes to audio samples through the codec (streaming state kept).
    fn decode_frame(&self, codes: &[usize; CODEBOOKS], s: &mut CodecState) -> Vec<f32> {
        let mut q = Vec::with_capacity(CODEBOOKS * 4);
        for &c in codes {
            q.extend(&self.fsq[c * 4..][..4]);
        }
        let z = self.fsq_out.0.forward_bias(&q, &self.fsq_out.1);
        let z = self.codec_in.0.forward_bias(&z, &self.codec_in.1);
        let mut x: Vec<f32> = (0..FRAME_STACK).flat_map(|_| z.iter().copied()).collect();
        for (b, st) in self.codec.iter().zip(&mut s.blocks) {
            b.forward(&mut x, st, true);
        }
        rms_norm(&mut x, Some(&self.codec_norm), 512, EPS);
        let polar = self.codec_out.0.forward_bias(&x, &self.codec_out.1);
        let bins = N_FFT / 2 + 1;
        let mut pcm = Vec::new();
        for frame in polar.chunks_exact(2 * bins) {
            let mut spec = vec![0f32; 2 * bins];
            for j in 0..bins {
                let mag = frame[j].exp();
                spec[2 * j] = mag * frame[bins + j].cos();
                spec[2 * j + 1] = mag * frame[bins + j].sin();
            }
            pcm.extend(s.istft.process(&spec));
        }
        pcm
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

struct CodecState {
    blocks: Vec<State>,
    istft: Istft,
}
impl CodecState {
    fn new(n: usize) -> Self {
        Self {
            blocks: vec![State::default(); n],
            istft: Istft::new(N_FFT, HOP),
        }
    }
}

/// A small deterministic generator (SplitMix64) for the codebook order and sampling.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
    /// Codes scoring at least the k-th best (ties kept), softmax at `temp`.
    fn sample_top_k(&mut self, l: &[f32], k: usize, temp: f32) -> usize {
        let mut sorted = l.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let lmin = sorted[k.min(sorted.len()) - 1];
        let lmax = sorted[0];
        let probs: Vec<f64> = l
            .iter()
            .map(|v| {
                if *v < lmin {
                    0.0
                } else {
                    (f64::from(v - lmax) / f64::from(temp)).exp()
                }
            })
            .collect();
        let mut r = self.uniform() * probs.iter().sum::<f64>();
        for (i, p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        l.len() - 1
    }
}

// ---- Audio frontend ---------------------------------------------------------------------

/// Slaney mel filters `[mel][bin]` (librosa's default, area-normalized).
fn mel_filterbank(n_mel: usize, n_fft: usize, sr: usize) -> Vec<f32> {
    let hz_to_mel = |f: f64| {
        if f < 1000.0 {
            f * 3.0 / 200.0
        } else {
            15.0 + (f / 1000.0).ln() / (6.4f64.ln() / 27.0)
        }
    };
    let mel_to_hz = |m: f64| {
        if m < 15.0 {
            m * 200.0 / 3.0
        } else {
            1000.0 * ((m - 15.0) * (6.4f64.ln() / 27.0)).exp()
        }
    };
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr as f64 / 2.0));
    let hz: Vec<f64> = (0..n_mel + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mel + 1) as f64))
        .collect();
    let bins = n_fft / 2 + 1;
    let step = sr as f64 / n_fft as f64;
    let mut out = vec![0f32; n_mel * bins];
    for m in 0..n_mel {
        let (l, c, r) = (hz[m], hz[m + 1], hz[m + 2]);
        let norm = 2.0 / (r - l).max(1e-30);
        for k in 0..bins {
            let f = k as f64 * step;
            let w = if f >= l && f <= c {
                (f - l) / (c - l).max(1e-30)
            } else if f > c && f <= r {
                (r - f) / (r - c).max(1e-30)
            } else {
                0.0
            };
            out[m * bins + k] = (w * norm) as f32;
        }
    }
    out
}

/// `[mel][frame]` log-mel rows of `samples` (padded on the right to whole hops) and the
/// frame count.
fn log_mel(samples: &[f32], filters: &[f32]) -> (Vec<f32>, usize) {
    let (win, hop, n_mel, bins) = (400usize, 160usize, 128usize, 201usize);
    let n = if samples.len() > win {
        samples.len() + (hop - (samples.len() - win) % hop) % hop
    } else {
        win
    };
    let mut padded = samples.to_vec();
    padded.resize(n, 0.0);
    let frames = (n - win) / hop + 1;
    let hann: Vec<f64> = (0..win)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / win as f64).cos()))
        .collect();
    let (cos, sin): (Vec<f64>, Vec<f64>) = (0..win)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / win as f64;
            (a.cos(), a.sin())
        })
        .unzip();
    use rayon::prelude::*;
    let power: Vec<Vec<f64>> = (0..frames)
        .into_par_iter()
        .map(|fr| {
            let x: Vec<f64> = (0..win)
                .map(|i| f64::from(padded[fr * hop + i]) * hann[i])
                .collect();
            (0..bins)
                .map(|k| {
                    let (mut re, mut im) = (0.0, 0.0);
                    for (i, v) in x.iter().enumerate() {
                        let idx = (k * i) % win;
                        re += v * cos[idx];
                        im -= v * sin[idx];
                    }
                    re * re + im * im
                })
                .collect()
        })
        .collect();
    let mut mel = vec![0f32; n_mel * frames];
    for (fr, p) in power.iter().enumerate() {
        for m in 0..n_mel {
            let sum: f64 = (0..bins)
                .map(|k| p[k] * f64::from(filters[m * bins + k]))
                .sum();
            mel[m * frames + fr] = ((sum.max(1e-12).log10() + 4.0) / 4.0) as f32;
        }
    }
    (mel, frames)
}

/// `[mel][time]` to one channel `[time][freq]`.
fn mel_to_tf(mel: &[f32], frames: usize) -> Vec<f32> {
    let mut x = vec![0f32; frames * 128];
    for m in 0..128 {
        for t in 0..frames {
            x[t * 128 + m] = mel[m * frames + t];
        }
    }
    x
}

/// A 3x3 stride-2 convolution over `[ch][time][freq]`: a zero column before the clip (the
/// streaming state at its start) and one after, freq padded by one on each side.
fn conv_down(x: &[f32], ch: usize, t: usize, f: usize, c: &Conv2d) -> (Vec<f32>, usize, usize) {
    use rayon::prelude::*;
    let tp = t + 2;
    let t2 = (tp - 3) / 2 + 1;
    let f2 = (f + 2 - 3) / 2 + 1;
    let at = |ci: usize, ti: isize, fi: isize| -> f32 {
        // Padded time index ti: 0 is the state column, 1..=t the clip, t+1 the right pad.
        if ti < 1 || ti as usize > t || fi < 0 || fi as usize >= f {
            0.0
        } else {
            x[(ci * t + ti as usize - 1) * f + fi as usize]
        }
    };
    let mut y = vec![0f32; c.cout * t2 * f2];
    y.par_chunks_exact_mut(t2 * f2)
        .enumerate()
        .for_each(|(co, out)| {
            for to in 0..t2 {
                for fo in 0..f2 {
                    let mut sum = c.b[co];
                    let inputs: Vec<usize> = if c.depthwise {
                        vec![co]
                    } else {
                        (0..ch).collect()
                    };
                    for (k, ci) in inputs.into_iter().enumerate() {
                        let wbase = if c.depthwise {
                            co * 9
                        } else {
                            (co * c.cin.max(ch) + k) * 9
                        };
                        for kf in 0..3 {
                            for kt in 0..3 {
                                let w = c.w[wbase + kf * 3 + kt];
                                sum +=
                                    w * at(ci, (to * 2 + kt) as isize, (fo * 2 + kf) as isize - 1);
                            }
                        }
                    }
                    out[to * f2 + fo] = sum;
                }
            }
        });
    (y, t2, f2)
}

/// A 1x1 convolution over `[ch][cells]`.
fn pointwise(x: &[f32], ch: usize, cells: usize, c: &Conv2d) -> Vec<f32> {
    use rayon::prelude::*;
    let mut y = vec![0f32; c.cout * cells];
    y.par_chunks_exact_mut(cells)
        .enumerate()
        .for_each(|(co, out)| {
            out.iter_mut().for_each(|v| *v = c.b[co]);
            for ci in 0..ch {
                let w = c.w[co * ch + ci];
                for (o, v) in out.iter_mut().zip(&x[ci * cells..][..cells]) {
                    *o += w * v;
                }
            }
        });
    y
}

// ---- Inverse STFT -----------------------------------------------------------------------

/// Overlap-add inverse STFT with a periodic Hann window, normalized by the summed squared
/// window, the first `(n_fft - hop) / 2` samples dropped (a centered transform).
struct Istft {
    n: usize,
    hop: usize,
    window: Vec<f32>,
    overlap: Vec<f32>,
    weight: Vec<f32>,
    skip: usize,
    cos: Vec<f64>,
    sin: Vec<f64>,
}
impl Istft {
    fn new(n: usize, hop: usize) -> Self {
        let window = (0..n)
            .map(|i| {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())) as f32
            })
            .collect();
        let (cos, sin) = (0..n)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (a.cos(), a.sin())
            })
            .unzip();
        Self {
            n,
            hop,
            window,
            overlap: vec![0.0; n],
            weight: vec![0.0; n],
            skip: (n - hop) / 2,
            cos,
            sin,
        }
    }
    /// One frame's spectrum (`n / 2 + 1` complex bins, interleaved) to up to `hop` samples.
    fn process(&mut self, spec: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let (n, bins) = (self.n, self.n / 2 + 1);
        // Real inverse DFT of a Hermitian spectrum: x[t] = (1/n) sum_k X_k e^{+2 pi i k t / n}.
        let frame: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|t| {
                let mut sum = f64::from(spec[0]);
                for k in 1..bins {
                    let idx = (k * t) % n;
                    let (re, im) = (f64::from(spec[2 * k]), f64::from(spec[2 * k + 1]));
                    let term = re * self.cos[idx] - im * self.sin[idx];
                    sum += if k == bins - 1 && n % 2 == 0 {
                        term
                    } else {
                        2.0 * term
                    };
                }
                (sum / n as f64) as f32
            })
            .collect();
        for ((w, o), (win, f)) in self
            .weight
            .iter_mut()
            .zip(&mut self.overlap)
            .zip(self.window.iter().zip(&frame))
        {
            *w += win * win;
            *o += f * win;
        }
        let mut out: Vec<f32> = (0..self.hop)
            .map(|i| {
                if self.weight[i] > 1e-8 {
                    self.overlap[i] / self.weight[i]
                } else {
                    self.overlap[i]
                }
            })
            .collect();
        self.overlap.copy_within(self.hop.., 0);
        self.weight.copy_within(self.hop.., 0);
        let end = n - self.hop;
        self.overlap[end..].fill(0.0);
        self.weight[end..].fill(0.0);
        let drop = self.skip.min(out.len());
        self.skip -= drop;
        out.drain(..drop);
        out
    }
    /// The samples left after the last frame, less the centered transform's padding.
    fn flush(&mut self) -> Vec<f32> {
        let rest = self.n - self.hop;
        let mut out: Vec<f32> = (0..rest)
            .map(|i| {
                if self.weight[i] > 1e-8 {
                    self.overlap[i] / self.weight[i]
                } else {
                    self.overlap[i]
                }
            })
            .collect();
        let trim = (self.n - self.hop) / 2;
        out.truncate(out.len().saturating_sub(trim));
        out
    }
}
