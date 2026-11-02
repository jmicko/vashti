//! Fixed-public numeric quality check for the small-expert arithmetic.
//!
//! Runs fixed public texts (repository documents and source) through the GPU
//! prefill twice, once with int8 small experts and once with the original f16
//! path, and a short text through the CPU f32 reference model. Prompt hidden
//! states are captured in RAM only; output distributions are computed on the
//! CPU and only aggregate statistics are printed (no tokens, no text).
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use cubecl_runtime::server::Handle;
use kuno_text::model::Qwen4Exp;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) const HC: usize = 10240;
const D: usize = 2560;
const VOCAB: usize = 248320;
/// CPU f32 reference prefix length (it runs at a few seconds per token).
pub(crate) const REFERENCE_TOKENS: usize = 192;

static ACTIVE: AtomicBool = AtomicBool::new(false);
/// (position, consumed token, unnormalized HC carry)
static ROWS: Mutex<Vec<(usize, u32, Vec<f32>)>> = Mutex::new(Vec::new());

pub(crate) fn requested() -> bool {
    matches!(std::env::var("KUNO_CLAUDE_QUALITY").as_deref(), Ok("1" | "chunk"))
}
/// `KUNO_CLAUDE_QUALITY=chunk`: one long public text, first as single prompt
/// chunks (report column "int8"), then capped at 4096-row chunks ("f16").
pub(crate) fn chunk_mode() -> bool {
    std::env::var("KUNO_CLAUDE_QUALITY").as_deref() == Ok("chunk")
}
/// Nonzero: cap prompt chunks at this many rows (chunk-mode second pass).
pub(crate) static CHUNK_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(crate) fn activate() {
    ACTIVE.store(true, Ordering::Relaxed);
}
pub(crate) fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

pub(crate) struct Text {
    pub name: &'static str,
    pub bytes: &'static [u8],
    /// Also evaluate the CPU f32 reference model (slow; short texts only).
    pub reference: bool,
}
fn cut(s: &'static str, n: usize) -> &'static [u8] {
    let mut n = n.min(s.len());
    while !s.is_char_boundary(n) {
        n -= 1;
    }
    &s.as_bytes()[..n]
}
pub(crate) fn texts() -> Vec<Text> {
    if chunk_mode() {
        return vec![Text { name: "long_spec", bytes: cut(include_str!("../../../../../docs/kuno-native-text-checkpoint.md"), 15000), reference: false }];
    }
    vec![
        Text { name: "prose_spec", bytes: cut(include_str!("../../../../../docs/vashti-images-spec.md"), 12000), reference: false },
        Text { name: "rust_source", bytes: cut(include_str!("claude_expert_regular.rs"), 9000), reference: false },
        Text { name: "prose_short", bytes: cut(include_str!("../../../../../docs/kuno-text-engine-research.md"), 1500), reference: true },
    ]
}

/// Called after a prompt chunk's last layer: rows start..start+count of the
/// final residual (f32, HC wide). RAM only.
pub(crate) fn capture(client: &ComputeClient<WgpuRuntime>, residual: &Handle, start: usize, tokens: &[u32]) -> kuno_text::Result<()> {
    let row_bytes = (HC * 4) as u64;
    let size = residual.size_in_used();
    eprintln!("public_quality_capture {}", serde_json::json!({"start":start,"count":tokens.len(),"buffer_rows":size / row_bytes}));
    if tokens.len() as u64 * row_bytes > size {
        return Err(kuno_text::Error::InvalidBufferLength);
    }
    let mut rows = ROWS.lock().unwrap();
    for (c, piece) in tokens.chunks(256).enumerate() {
        // Chunk mode: positions >= 2048 only (the private I/O budget is ~4K rows).
        if chunk_mode() && start + c * 256 + piece.len() <= 2048 {
            continue;
        }
        let first = (c * 256) as u64 * row_bytes;
        let bytes = piece.len() as u64 * row_bytes;
        let data = client
            .read_one(residual.clone().offset_start(first).offset_end(size - first - bytes))
            .map_err(|e| {
                eprintln!("public_quality_capture_read_failed piece={c} {}", e.to_string().replace('\n', " | "));
                kuno_text::Error::InvalidBufferLength
            })?;
        for (i, &t) in piece.iter().enumerate() {
            let row = data[i * HC * 4..(i + 1) * HC * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            rows.push((start + c * 256 + i, t, row));
        }
    }
    Ok(())
}
pub(crate) fn take() -> Vec<(usize, u32, Vec<f32>)> {
    std::mem::take(&mut *ROWS.lock().unwrap())
}

/// Batched CPU output head: logits[v] = output.weight · mixed[v].
fn head(graph: &Qwen4Exp<'_>, mixed: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let tensor = graph.bound_graph().tensor("output.weight").expect("output head");
    let m = mixed.len();
    let threads = 32usize;
    let per = VOCAB.div_ceil(threads);
    let mut parts: Vec<Vec<f32>> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let (r0, r1) = (t * per, ((t + 1) * per).min(VOCAB));
                    let mut row = vec![0f32; D];
                    let mut out = vec![0f32; (r1 - r0) * m];
                    for r in r0..r1 {
                        tensor.decode_row(r as u64, &mut row).expect("head row");
                        for (v, x) in mixed.iter().enumerate() {
                            out[(r - r0) * m + v] = dot(&row, x);
                        }
                    }
                    out
                })
            })
            .collect();
        parts = handles.into_iter().map(|h| h.join().unwrap()).collect();
    });
    let mut logits = vec![vec![0f32; VOCAB]; m];
    for (t, part) in parts.iter().enumerate() {
        let r0 = t * per;
        for (i, chunk) in part.chunks_exact(m).enumerate() {
            for v in 0..m {
                logits[v][r0 + i] = chunk[v];
            }
        }
    }
    logits
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    for (x, y) in a.chunks_exact(16).zip(b.chunks_exact(16)) {
        for k in 0..16 {
            acc[k] = x[k].mul_add(y[k], acc[k]);
        }
    }
    acc.iter().sum()
}
fn log_softmax(l: &[f32]) -> Vec<f64> {
    let max = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let sum: f64 = l.iter().map(|&x| (x as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    l.iter().map(|&x| x as f64 - lse).collect()
}
fn argmax(l: &[f32]) -> usize {
    l.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0
}

/// Running comparison of a candidate distribution against a baseline.
#[derive(Default)]
struct Compare {
    kl: Vec<f64>,
    top1: usize,
    nll_base: f64,
    nll_cand: f64,
    hidden_rel: Vec<f64>,
}
impl Compare {
    fn add(&mut self, base: &[f32], cand: &[f32], next: usize) {
        let (p, q) = (log_softmax(base), log_softmax(cand));
        let kl: f64 = p.iter().zip(&q).map(|(&a, &b)| a.exp() * (a - b)).sum();
        self.kl.push(kl.max(0.));
        self.top1 += usize::from(argmax(base) == argmax(cand));
        self.nll_base -= p[next];
        self.nll_cand -= q[next];
    }
    fn json(&mut self) -> serde_json::Value {
        let n = self.kl.len().max(1);
        self.kl.sort_by(|a, b| a.total_cmp(b));
        let pct = |v: &[f64], f: f64| v.get(((v.len() as f64 - 1.) * f).round() as usize).copied().unwrap_or(0.);
        self.hidden_rel.sort_by(|a, b| a.total_cmp(b));
        serde_json::json!({
            "positions": self.kl.len(),
            "kl_mean": self.kl.iter().sum::<f64>() / n as f64,
            "kl_median": pct(&self.kl, 0.5), "kl_p99": pct(&self.kl, 0.99), "kl_max": pct(&self.kl, 1.),
            "top1_agree": self.top1 as f64 / n as f64,
            "ppl_base": (self.nll_base / n as f64).exp(), "ppl_cand": (self.nll_cand / n as f64).exp(),
            "hidden_rel_mean": self.hidden_rel.iter().sum::<f64>() / self.hidden_rel.len().max(1) as f64,
            "hidden_rel_max": pct(&self.hidden_rel, 1.),
        })
    }
}
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (x as f64 - y as f64).powi(2);
        n += (y as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

/// int8 vs f16 rows of one text (both full prompt captures, sorted by position).
/// `reference` = CPU f32 logits per position (all positions), when evaluated.
pub(crate) fn report(
    graph: &Qwen4Exp<'_>,
    name: &str,
    int8: &[(usize, u32, Vec<f32>)],
    f16: &[(usize, u32, Vec<f32>)],
    reference: Option<&[(usize, Vec<f32>)]>,
) -> kuno_text::Result<()> {
    if int8.len() != f16.len() || int8.iter().zip(f16).any(|(a, b)| a.0 != b.0 || a.1 != b.1) {
        eprintln!("public_quality {}", serde_json::json!({"text":name,"error":"capture mismatch","int8_rows":int8.len(),"f16_rows":f16.len()}));
        return Ok(());
    }
    let started = std::time::Instant::now();
    let cancel = AtomicBool::new(false);
    let mut exec = kuno_text::threaded::ThreadedExecutor::new(32)?;
    // Position i predicts the token consumed at i+1.
    let stride = if chunk_mode() { 1 } else { (int8.len() / 1024).max(1) };
    let picks: Vec<usize> = (0..int8.len().saturating_sub(1)).step_by(stride).collect();
    let mut f16_vs_i8 = Compare::default();
    // Chunk mode: positions past the old 4096-row chunk boundary, separately.
    let mut late = Compare::default();
    let (mut ref_vs_i8, mut ref_vs_f16) = (Compare::default(), Compare::default());
    for &i in &picks {
        f16_vs_i8.hidden_rel.push(rel(&int8[i].2, &f16[i].2));
    }
    for block in picks.chunks(64) {
        let mut mixed = Vec::with_capacity(block.len() * 2);
        for &i in block {
            mixed.push(graph.output_mixed(&mut exec, &int8[i].2, &cancel)?);
            mixed.push(graph.output_mixed(&mut exec, &f16[i].2, &cancel)?);
        }
        let logits = head(graph, &mixed);
        for (k, &i) in block.iter().enumerate() {
            let next = int8[i + 1].1 as usize;
            let (a, b) = (&logits[2 * k], &logits[2 * k + 1]);
            f16_vs_i8.add(b, a, next);
            if int8[i].0 >= 4096 {
                late.hidden_rel.push(rel(&int8[i].2, &f16[i].2));
                late.add(b, a, next);
            }
            if let Some(r) = reference.and_then(|r| r.iter().find(|x| x.0 == int8[i].0)) {
                ref_vs_i8.add(&r.1, a, next);
                ref_vs_f16.add(&r.1, b, next);
            }
        }
    }
    let mut out = serde_json::json!({
        "text": name, "tokens": int8.len(), "stride": stride, "seconds": started.elapsed().as_secs_f64(),
        "f16_base_int8_cand": f16_vs_i8.json(),
    });
    if chunk_mode() {
        out["positions_from_4096"] = late.json();
    }
    if reference.is_some() {
        out["cpu_f32_base_int8_cand"] = ref_vs_i8.json();
        out["cpu_f32_base_f16_cand"] = ref_vs_f16.json();
    }
    eprintln!("public_quality {out}");
    Ok(())
}

/// CPU f32 reference logits for every prompt position of `tokens`.
pub(crate) fn reference(graph: &Qwen4Exp<'_>, tokens: &[u32]) -> kuno_text::Result<Vec<(usize, Vec<f32>)>> {
    let started = std::time::Instant::now();
    let cancel = AtomicBool::new(false);
    let mut exec = kuno_text::threaded::ThreadedExecutor::new(32)?;
    let mut seq = graph.sequence(tokens.len() + 8)?;
    let mut out = Vec::with_capacity(tokens.len());
    for (p, &t) in tokens.iter().enumerate() {
        out.push((p, graph.step(&mut exec, &mut seq, t, &cancel, |_| {})?));
        if p % 16 == 15 {
            eprintln!("public_quality_reference_rate {}", serde_json::json!({"tokens":p + 1,"seconds":started.elapsed().as_secs_f64()}));
        }
    }
    eprintln!("public_quality_reference_done {}", serde_json::json!({"tokens":tokens.len(),"seconds":started.elapsed().as_secs_f64()}));
    Ok(out)
}
