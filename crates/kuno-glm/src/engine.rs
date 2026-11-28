//! Decoding one token at a time on the GPU, the routed experts on the CPU: each MoE block's
//! work up to the router is one submission; the host reads the chosen experts, runs them
//! (`cpu`), and the next submission adds their sum while it applies the block's mixer.
use kuno_image::engine::gpu::{Buffer, Gpu, Kernel, Mapped, Passing, Step, bytes_of};

use crate::experts::{self, BlockExperts};
use crate::gpu::{Kernels, Mat, filled};
use crate::{Matrix, Model, Result};

mod batch;
mod pipe;
mod spec;
pub use spec::SpecStats;
pub use spec::uses_report;

/// The engine's kernels besides the products.
struct Extra {
    hcdot: Kernel,
    hcmix: Kernel,
    kda: Kernel,
    dsanorm: Kernel,
    mla: Kernel,
    route: Kernel,
    glu: Kernel,
    fin: Kernel,
    argmax: Kernel,
    mve: Kernel,
    idxstore: Kernel,
    idxscore: Kernel,
    idxselect: Kernel,
    copyoff: Kernel,
    /// The IQ2_S and IQ3_S grids for mve.
    tables: Buffer,
    /// Batched variants (prompt processing, `batch`).
    b: Batched,
}

struct Batched {
    hcdot: Kernel,
    hcmix: Kernel,
    kda: Kernel,
    dsanorm: Kernel,
    idxstore: Kernel,
    idxscore: Kernel,
    idxselect: Kernel,
    mla: Kernel,
    route: Kernel,
    mm: Kernel,
    /// mv for up to 4 tokens (a verification's batch), sums as mv's.
    mvt: Kernel,
    /// Q6_K for up to 4 tokens, four rows per subgroup (the inputs read a quarter as often).
    mvr: Kernel,
    /// mvr with two rows per subgroup (matrices of few rows).
    mvr2: Kernel,
    /// The cooperative-matrix products (Intel Xe2's half-precision shape): dense and experts.
    /// Without them a prompt runs a token at a time.
    mmc: Option<Kernel>,
    mec: Option<Kernel>,
    group: Kernel,
    combine: Kernel,
    copy4: Kernel,
    rmsb: Kernel,
    sum3: Kernel,
    mlac: Kernel,
}

struct Mixer {
    function: Mat,
    /// base [24], scale [3]
    params: Buffer,
    norm: Buffer,
}

#[allow(clippy::large_enum_variant)]
enum Attention {
    Kda {
        qkv: Mat,
        /// f_a, g_a, beta rows
        small: Mat,
        /// f_b then g_b rows (per-group inputs: f_a's 128, g_a's 128)
        fg_b: Mat,
        conv: Buffer,
        /// a [64], out norm [128], dt bias [d]
        params: Buffer,
        out: Mat,
        hist: Buffer,
        state: Buffer,
    },
    Dsa {
        q_a: Mat,
        kv_a: Mat,
        norms: Buffer,
        q_b: Mat,
        k_b: Mat,
        v_b: Mat,
        out: Mat,
        latents: Buffer,
        /// The indexer: key|gate rows, head weights (f32), query rows; LayerNorm weight,
        /// bias and the pooling's position bias; the key|gate cache and pooled keys.
        idx_kg: Mat,
        idx_proj: Mat,
        idx_qb: Mat,
        idx_params: Buffer,
        idx_keys: Buffer,
        idx_pooled: Buffer,
    },
}

enum Ffn {
    Dense {
        gu: Mat,
        down: Mat,
    },
    Moe {
        router: Mat,
        bias: Buffer,
        gu: Mat,
        down: Mat,
    },
}

struct Block {
    attn: Mixer,
    ffn_mixer: Mixer,
    attention: Attention,
    ffn: Ffn,
}

/// Device scratch for `batch` tokens (one token's at the start of each buffer while decoding).
struct Scratch {
    /// The four streams, alternating between the two.
    x: [Buffer; 2],
    part: Buffer,
    coef: Buffer,
    normed: Buffer,
    /// Projections (the largest is DSA's q_b, 16384, or KDA's qkv, 24576).
    a: Buffer,
    b: Buffer,
    c: Buffer,
    out: Buffer,
    logits: Buffer,
    /// The device's experts: gate|up per chosen expert, their SwiGLU, the weighted down sum.
    ge: Buffer,
    gh: Buffer,
    gd: Buffer,
    zeros: Buffer,
    /// The router's choice in device memory (ids, weights).
    routed: Buffer,
    /// The indexer: key|gate and head weights; its query; pool scores; chosen positions.
    ia: Buffer,
    ib: Buffer,
    iscore: Buffer,
    iidx: Buffer,
    /// A batch's routing grouped by expert (`group.comp`).
    lists: Buffer,
}

pub struct Engine<'m> {
    model: &'m Model,
    gpu: Gpu,
    k: Kernels,
    e: Extra,
    blocks: Vec<Block>,
    /// Per block its experts, once loaded: up front with a second device, else on the way
    /// (a prompt's batch loads each block as it reaches it; `load_all` the rest).
    experts: Vec<std::sync::OnceLock<BlockExperts>>,
    /// Device bank slots per block (for blocks loaded later).
    slots1: usize,
    second: Option<crate::second::Second>,
    out_norm: Buffer,
    output: Mat,
    embed: Matrix,
    s: Scratch,
    /// Host-visible: the embedding x4, the router's choice (+ normed input), the CPU's
    /// expert sum, the next token.
    host_x: Mapped,
    route: Mapped,
    cpu_out: Mapped,
    token: Mapped,
    pos: usize,
    max_positions: usize,
    /// Tokens per batch while processing a prompt (1: one at a time).
    batch: usize,
    /// Per MoE block, its sources' expert order for mme (device slots, then host slots); how
    /// often a batch chose each expert per block.
    order: Mapped,
    usage: Mapped,
    /// Two device buffers the host experts stream through while a batch runs (`piece` slots
    /// each; empty without batching).
    stage: Vec<Buffer>,
    piece: usize,
    /// Per block (KDA blocks, with speculation): its state and conv history after each of
    /// a verification's first tokens (`attention_ops`' snapshots).
    snap: Vec<Option<Buffer>>,
    /// Products in f32 only (no half-precision cooperative matrices): a verification's
    /// sums then agree with `step`'s.
    exact: std::sync::atomic::AtomicBool,
    /// Speculation (KUNO_GLM_SPEC drafts per step): the NextN block; the main model's h at
    /// every position of the last batch [batch][n]; a small batch's logits, routing and
    /// inputs for the CPU, and the CPU's sums; drafts per step at most (snapshots).
    mtp: Option<spec::Mtp>,
    /// Attention's partial results for small batches (mlab.comp's runs).
    mpart: Buffer,
    hall: Buffer,
    logits_t: Buffer,
    vin: Mapped,
    cpu_t: Mapped,
    snaps: usize,
    /// Per MoE block (with speculation): the slot in its staging buffer of each expert the
    /// device takes from a verification's host experts.
    gslots: Vec<Mapped>,
    /// The block the next speculative trade round starts from.
    trade_turn: std::sync::atomic::AtomicUsize,
    /// With a side queue: the staging buffers are a ring the side queue fills ahead of the
    /// main queue; `copied` counts pieces copied, `consumed` pieces used, `streamed` pieces
    /// so far (the timelines' next values).
    copied: Option<kuno_image::engine::gpu::Timeline>,
    consumed: Option<kuno_image::engine::gpu::Timeline>,
    streamed: std::sync::atomic::AtomicU64,
    /// When set, the mean of the streams after each block (comparison with the reference).
    pub trace: Option<Vec<Vec<f32>>>,
    /// Staged slot maps for trades, and this token's trades.
    trade_stage: Vec<Mapped>,
    trades: std::sync::Mutex<Vec<experts::Trade>>,
    /// With KUNO_GLM_PROFILE: device time per kernel (and product shape), ns and count.
    pub profile: std::sync::Mutex<std::collections::BTreeMap<String, (f64, u32)>>,
    /// Lanes for pipelined verifications (tokens after the first; the first is `step`'s).
    lanes: Vec<LaneBufs>,
    lane_maps: Vec<Mapped>,
    /// Seconds in the CPU's sums and in waiting for the second device, this token.
    split: std::sync::Mutex<[f64; 2]>,
}

fn concat(ms: &[Matrix]) -> Matrix {
    let mut bytes = Vec::new();
    for m in ms {
        assert_eq!((m.dtype, m.cols), (ms[0].dtype, ms[0].cols));
        bytes.extend_from_slice(&m.bytes);
    }
    Matrix {
        dtype: ms[0].dtype,
        rows: ms.iter().map(|m| m.rows).sum(),
        cols: ms[0].cols,
        row_bytes: ms[0].row_bytes,
        bytes,
    }
}

fn floats(gpu: &Gpu, v: &[f32]) -> Result<Buffer> {
    filled(gpu, bytes_of(v))
}

/// One recorded step: owned buffers list and constants, borrowed by `Step` at submission.
struct Op<'a> {
    kernel: &'a Kernel,
    buffers: Vec<&'a Buffer>,
    constants: Vec<u32>,
    groups: [u32; 3],
    /// No barrier after this step (the next one does not read its output).
    independent: bool,
}

fn f(v: f32) -> u32 {
    v.to_bits()
}

/// The router's sequence word for block `il` at position `pos` (never 0, the buffer's start).
fn seq(pos: usize, il: usize) -> u32 {
    (pos as u32).wrapping_mul(64).wrapping_add(il as u32 + 1) | 1 << 31
}

impl<'m> Engine<'m> {
    /// `second`: a further GPU for the next tier of experts.
    pub fn new(
        model: &'m Model,
        gpu: Gpu,
        second: Option<Gpu>,
        max_positions: usize,
    ) -> Result<Self> {
        let hp = &model.hp;
        let loading = std::time::Instant::now();
        let k = Kernels::new(&gpu)?;
        let lanes = Some(k.lanes);
        let kernel = |spirv: &[u8], b, c| gpu.kernel(spirv, b, c, lanes);
        let e = Extra {
            hcdot: kernel(include_bytes!("../shaders/hcdot.spv"), 9, 3)?,
            hcmix: kernel(include_bytes!("../shaders/hcmix.spv"), 6, 5)?,
            kda: kernel(include_bytes!("../shaders/kda.spv"), 8, 4)?,
            dsanorm: kernel(include_bytes!("../shaders/dsanorm.spv"), 4, 4)?,
            mla: kernel(include_bytes!("../shaders/mla.spv"), 4, 4)?,
            idxstore: kernel(include_bytes!("../shaders/idxstore.spv"), 4, 2)?,
            idxscore: kernel(include_bytes!("../shaders/idxscore.spv"), 4, 3)?,
            idxselect: gpu.kernel(include_bytes!("../shaders/idxselect.spv"), 2, 4, None)?,
            copyoff: kernel(include_bytes!("../shaders/copyoff.spv"), 2, 3)?,
            route: kernel(include_bytes!("../shaders/route.spv"), 5, 5)?,
            glu: kernel(include_bytes!("../shaders/glu.spv"), 2, 3)?,
            fin: kernel(include_bytes!("../shaders/final.spv"), 3, 2)?,
            argmax: gpu.kernel(include_bytes!("../shaders/argmax.spv"), 2, 2, None)?,
            mve: kernel(include_bytes!("../shaders/mve.spv"), 7, 10)?,
            tables: {
                let mut t: Vec<u32> = Vec::with_capacity(2560);
                for g in kuno_text::quant::IQ2_S_GRID {
                    t.push(g as u32);
                    t.push((g >> 32) as u32);
                }
                t.extend(kuno_text::quant::IQ3_S_GRID);
                filled(&gpu, bytes_of(&t))?
            },
            b: Batched {
                hcdot: kernel(include_bytes!("../shaders/hcdot_b.spv"), 9, 6)?,
                hcmix: kernel(include_bytes!("../shaders/hcmix_b.spv"), 6, 5)?,
                kda: kernel(include_bytes!("../shaders/kdab.spv"), 9, 10)?,
                dsanorm: kernel(include_bytes!("../shaders/dsanorm_b.spv"), 4, 6)?,
                idxstore: kernel(include_bytes!("../shaders/idxstore_b.spv"), 4, 4)?,
                idxscore: kernel(include_bytes!("../shaders/idxscore_b.spv"), 4, 8)?,
                idxselect: gpu.kernel(include_bytes!("../shaders/idxselect_b.spv"), 2, 5, None)?,
                mla: kernel(include_bytes!("../shaders/mlab.spv"), 5, 8)?,
                mlac: kernel(include_bytes!("../shaders/mlac.spv"), 2, 4)?,
                route: kernel(include_bytes!("../shaders/route_b.spv"), 5, 3)?,
                mm: kernel(include_bytes!("../shaders/mm.spv"), 4, 12)?,
                mvt: kernel(include_bytes!("../shaders/mvt.spv"), 4, 12)?,
                mvr: kernel(include_bytes!("../shaders/mvr.spv"), 4, 12)?,
                mvr2: kernel(include_bytes!("../shaders/mvr2.spv"), 4, 12)?,
                mmc: if gpu.has_half8() && k.lanes == 16 {
                    Some(kernel(include_bytes!("../shaders/mmc.spv"), 4, 10)?)
                } else {
                    None
                },
                mec: if gpu.has_half8() && k.lanes == 16 {
                    Some(kernel(include_bytes!("../shaders/mec.spv"), 7, 12)?)
                } else {
                    None
                },
                group: gpu.kernel(include_bytes!("../shaders/group.spv"), 3, 5, None)?,
                combine: kernel(include_bytes!("../shaders/combine.spv"), 5, 6)?,
                copy4: kernel(include_bytes!("../shaders/copy4.spv"), 2, 3)?,
                rmsb: kernel(include_bytes!("../shaders/rmsb.spv"), 3, 6)?,
                sum3: kernel(include_bytes!("../shaders/sum3.spv"), 4, 6)?,
            },
        };

        let reader = model.reader()?;
        // Every weight but the experts (and the unused NextN block) read up front, in
        // parallel at the disk's speed, then taken as the blocks are built.
        let pre: std::sync::Mutex<std::collections::HashMap<String, Matrix>> = {
            use rayon::prelude::*;
            let skip = format!("blk.{}.", hp.layers);
            let names: Vec<&String> = model
                .gguf
                .tensors()
                .keys()
                .filter(|n| !n.contains("_exps") && !n.starts_with(&skip))
                .collect();
            let read: Result<std::collections::HashMap<String, Matrix>> = names
                .par_iter()
                .map(|n| Ok(((*n).clone(), model.matrix_with(&reader, n)?)))
                .collect();
            std::sync::Mutex::new(read?)
        };
        eprintln!("weights read ({:.1} s)", loading.elapsed().as_secs_f64());
        let m = |n: &str| -> Result<Matrix> {
            match pre.lock().map_err(|_| "model_read_failed")?.remove(n) {
                Some(x) => Ok(x),
                None => model.matrix_with(&reader, n),
            }
        };
        let v = |n: &str| m(n)?.decode_all();
        let up = |mx: &Matrix| Mat::upload(&gpu, mx);
        // A DSA block's attention (also the NextN block's).
        let dsa = |il: usize| -> Result<Attention> {
            let b = |n: &str| format!("blk.{il}.{n}");
            let mut norms = v(&b("attn_q_a_norm.weight"))?;
            norms.extend(v(&b("attn_kv_a_norm.weight"))?);
            let attention = Attention::Dsa {
                q_a: up(&m(&b("attn_q_a.weight"))?)?,
                kv_a: up(&m(&b("attn_kv_a_mqa.weight"))?)?,
                norms: floats(&gpu, &norms)?,
                q_b: up(&m(&b("attn_q_b.weight"))?)?,
                k_b: up(&m(&b("attn_k_b.weight"))?)?,
                v_b: up(&m(&b("attn_v_b.weight"))?)?,
                out: up(&m(&b("attn_output.weight"))?)?,
                latents: gpu.buffer((max_positions * hp.kv_lora * 4) as u64)?,
                idx_kg: up(&concat(&[
                    m(&b("indexer.attn_k.weight"))?,
                    m(&b("indexer_compressor_gate.weight"))?,
                ]))?,
                idx_proj: up(&m(&b("indexer.proj.weight"))?)?,
                idx_qb: up(&m(&b("indexer.attn_q_b.weight"))?)?,
                idx_params: {
                    let mut p = v(&b("indexer.k_norm.weight"))?;
                    p.extend(v(&b("indexer.k_norm.bias"))?);
                    p.extend(v(&b("indexer_compressor_ape.weight"))?);
                    floats(&gpu, &p)?
                },
                idx_keys: gpu.buffer((max_positions * 256 * 4) as u64)?,
                idx_pooled: gpu.buffer((max_positions / 4 * 128 * 4).max(16) as u64)?,
            };
            Ok(attention)
        };
        let mut blocks = Vec::with_capacity(hp.layers);
        for il in 0..hp.layers {
            let b = |n: &str| format!("blk.{il}.{n}");
            let mixer = |part: &str, norm: &str| -> Result<Mixer> {
                let mut params = v(&b(&format!("hc_{part}_base.weight")))?;
                params.extend(v(&b(&format!("hc_{part}_scale.weight")))?);
                Ok(Mixer {
                    function: up(&m(&b(&format!("hc_{part}_fn.weight")))?)?,
                    params: floats(&gpu, &params)?,
                    norm: floats(&gpu, &v(&b(norm))?)?,
                })
            };
            let attention = if hp.dsa[il] {
                dsa(il)?
            } else {
                let d = hp.kda_head * hp.heads;
                let mut conv = Vec::with_capacity(3 * d * 4);
                for part in ["q", "k", "v"] {
                    conv.extend(v(&b(&format!("ssm_conv1d_{part}.weight")))?);
                }
                let mut params = v(&b("ssm_a"))?;
                params.extend(v(&b("ssm_norm.weight"))?);
                params.extend(v(&b("ssm_dt.bias"))?);
                let zeros = |n: usize| floats(&gpu, &vec![0f32; n]);
                Attention::Kda {
                    qkv: up(&concat(&[
                        m(&b("attn_q.weight"))?,
                        m(&b("attn_k.weight"))?,
                        m(&b("attn_v.weight"))?,
                    ]))?,
                    small: up(&concat(&[
                        m(&b("ssm_f_a.weight"))?,
                        m(&b("ssm_g_a.weight"))?,
                        m(&b("ssm_beta.weight"))?,
                    ]))?,
                    fg_b: up(&concat(&[
                        m(&b("ssm_f_b.weight"))?,
                        m(&b("ssm_g_b.weight"))?,
                    ]))?,
                    conv: floats(&gpu, &conv)?,
                    params: floats(&gpu, &params)?,
                    out: up(&m(&b("attn_output.weight"))?)?,
                    hist: zeros(3 * 3 * d)?,
                    state: zeros(hp.heads * hp.kda_head * hp.kda_head)?,
                }
            };
            let ffn = if il < hp.dense_lead {
                Ffn::Dense {
                    gu: up(&concat(&[
                        m(&b("ffn_gate.weight"))?,
                        m(&b("ffn_up.weight"))?,
                    ]))?,
                    down: up(&m(&b("ffn_down.weight"))?)?,
                }
            } else {
                Ffn::Moe {
                    router: up(&m(&b("ffn_gate_inp.weight"))?)?,
                    bias: floats(&gpu, &v(&b("exp_probs_b.bias"))?)?,
                    gu: up(&concat(&[
                        m(&b("ffn_gate_shexp.weight"))?,
                        m(&b("ffn_up_shexp.weight"))?,
                    ]))?,
                    down: up(&m(&b("ffn_down_shexp.weight"))?)?,
                }
            };
            blocks.push(Block {
                attn: mixer("attn", "attn_norm.weight")?,
                ffn_mixer: mixer("ffn", "ffn_norm.weight")?,
                attention,
                ffn,
            });
            eprintln!(
                "block {il} on the device ({:.1} s)",
                loading.elapsed().as_secs_f64()
            );
        }
        let n = hp.embd;
        // KUNO_GLM_BATCH: tokens per batch of a prompt (default 2048, ~1 MB of device scratch
        // per token; 1 runs it a token at a time). The scratch below is per token, times it.
        let batch: usize = std::env::var("KUNO_GLM_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048usize)
            .clamp(1, 4096)
            .next_multiple_of(8);
        let used = hp.experts_used;
        let lp = batch * used + 8 * hp.experts;
        let s = Scratch::new(&gpu, hp, batch, lp, max_positions)?;
        let buf = |floats: usize| gpu.buffer((floats * batch * 4) as u64);
        let output = up(&m("output.weight")?)?;
        // Speculation: KUNO_GLM_SPEC drafts per step (default 2; 0 for none) with the NextN
        // block, when the file has it; per KDA block snapshots for that many tokens.
        let drafts: usize = std::env::var("KUNO_GLM_SPEC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2usize)
            .min(spec::SMALL - 1);
        let il = hp.layers;
        let has_mtp = model
            .gguf
            .tensor(&format!("blk.{il}.nextn.eh_proj.weight"))
            .is_some();
        // (The staging buffers exist with batching on cooperative matrices; see below.)
        let stage_empty = !(batch > 1 && e.b.mec.is_some());
        let mtp = if drafts > 0 && has_mtp && batch > 1 && second.is_none() {
            let b = |n: &str| format!("blk.{il}.{n}");
            let mut norms = v(&b("nextn.enorm.weight"))?;
            for name in [
                "nextn.hnorm.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
                "nextn.shared_head_norm.weight",
            ] {
                norms.extend(v(&b(name))?);
            }
            let (gk, gr, _) = model.layout(&b("ffn_gate_exps.weight"))?;
            let (dk, dr, _) = model.layout(&b("ffn_down_exps.weight"))?;
            Some(spec::Mtp {
                eh: up(&m(&b("nextn.eh_proj.weight"))?)?,
                norms: floats(&gpu, &norms)?,
                attention: dsa(il)?,
                router: up(&m(&b("ffn_gate_inp.weight"))?)?,
                bias: floats(&gpu, &v(&b("exp_probs_b.bias"))?)?,
                gu: up(&concat(&[
                    m(&b("ffn_gate_shexp.weight"))?,
                    m(&b("ffn_up_shexp.weight"))?,
                ]))?,
                down: up(&m(&b("ffn_down_shexp.weight"))?)?,
                kinds: (gk, dk),
                row_bytes: (gr, dr),
                experts: std::sync::OnceLock::new(),
                cat: buf(2 * n)?,
                res: buf(n)?,
                inp: buf(n)?,
                post: buf(n)?,
                h: buf(n)?,
                last: floats(&gpu, &vec![0f32; n])?,
                emb: gpu.mapped((batch * n * 4) as u64)?,
                vin: gpu.mapped(((2 * hp.experts_used + n) * 4) as u64)?,
                cpu: gpu.mapped((n * 4) as u64)?,
            })
        } else {
            None
        };
        let drafts = if mtp.is_some() { drafts } else { 0 };
        let mut snap = Vec::with_capacity(hp.layers);
        for il in 0..hp.layers {
            snap.push(if drafts > 0 && !hp.dsa[il] {
                let d = hp.kda_head * hp.heads;
                Some(gpu.buffer((drafts * (hp.heads * 128 * 128 + 9 * d) * 4) as u64)?)
            } else {
                None
            });
        }
        if mtp.is_some() {
            eprintln!("speculation: {drafts} drafts per step with the NextN block");
        }
        // The experts: the device's free memory less a margin (or KUNO_GLM_DEVICE_GB).
        let margin = 1u64 << 30;
        // The staging pair for batches: KUNO_GLM_PIECE slots each (default 24, ~0.2 GB).
        let piece: usize = std::env::var("KUNO_GLM_PIECE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24usize)
            .max(1);
        let mut stage = Vec::new();
        if batch > 1 && e.b.mec.is_some() {
            let mut slot_bytes = 0u64;
            for il in hp.dense_lead..hp.layers {
                let name = |p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
                let (_, gr, _) = model.layout(&name("gate"))?;
                let (_, dr, _) = model.layout(&name("down"))?;
                slot_bytes = slot_bytes.max((2 * hp.ff_expert * gr + n * dr) as u64);
            }
            // KUNO_GLM_RING staging buffers with a side queue (default 6: ~1.3 GB, about what
            // streams while a block's attention and dense products run), else two.
            let ring: usize = if gpu.has_side() {
                std::env::var("KUNO_GLM_RING")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(6usize)
                    .max(2)
            } else {
                2
            };
            for _ in 0..ring {
                stage.push(gpu.buffer(piece as u64 * slot_bytes)?);
            }
        }
        let budget = std::env::var("KUNO_GLM_DEVICE_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|g| (g * 1e9) as u64)
            .unwrap_or_else(|| gpu.free_memory().unwrap_or(0).saturating_sub(margin));
        let second = match second {
            Some(g) => Some(crate::second::Second::new(
                g,
                hp.experts_used,
                n,
                hp.ff_expert,
            )?),
            None => None,
        };
        let budget2 = second.as_ref().map(|s| {
            std::env::var("KUNO_GLM_SECOND_GB")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .map(|g| (g * 1e9) as u64)
                .unwrap_or_else(|| {
                    s.gpu
                        .free_memory()
                        .unwrap_or(0)
                        .saturating_sub(3 * margin / 2)
                })
        });
        // KUNO_GLM_LAZY=0 (or a second device, or KUNO_GLM_NO_HOST): every block's experts
        // now; else each when first needed.
        let lazy = second.is_none()
            && std::env::var_os("KUNO_GLM_NO_HOST").is_none()
            && !std::env::var("KUNO_GLM_LAZY").is_ok_and(|v| v == "0");
        let (slots1, _, per_slot) = experts::plan(model, budget, None)?;
        let experts: Vec<std::sync::OnceLock<BlockExperts>> = if lazy {
            eprintln!(
                "experts: {slots1} slots per block on the device ({:.1} GB), loaded on the way",
                (slots1 as u64 * per_slot) as f64 / 1e9
            );
            (0..hp.layers).map(|_| std::sync::OnceLock::new()).collect()
        } else {
            experts::place(
                model,
                &gpu,
                budget,
                second.as_ref().map(|s| (&s.gpu, budget2.unwrap_or(0))),
            )?
            .into_iter()
            .map(|b| {
                let c = std::sync::OnceLock::new();
                if let Some(b) = b {
                    let _ = c.set(b);
                }
                c
            })
            .collect()
        };
        // Trades per token (KUNO_GLM_TRADES, default one per block): each a staged slot map.
        let trade_stage = {
            let count = std::env::var("KUNO_GLM_TRADES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(hp.layers);
            let mut v = Vec::new();
            if slots1 > 0 {
                for _ in 0..count {
                    v.push(gpu.mapped((hp.experts * 4) as u64)?);
                }
            }
            v
        };
        // KUNO_GLM_PIPE=1: pipelined verifications (`pipe`), a lane per draft beside
        // `step`'s own (~100 MB each). Exactly `step`'s results, but slower than the batched
        // verification where the host's memory bandwidth bounds the CPU's experts.
        let lanes = if mtp.is_some() && std::env::var("KUNO_GLM_PIPE").is_ok_and(|v| v == "1") {
            (0..drafts)
                .map(|_| LaneBufs::new(&gpu, hp, max_positions))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        // Per lane (the first too), the staging slot of each expert the device takes.
        let lane_maps = if lanes.is_empty() {
            Vec::new()
        } else {
            (0..=drafts)
                .map(|_| gpu.mapped((hp.experts * 4) as u64))
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        Ok(Self {
            lanes,
            lane_maps,
            out_norm: floats(&gpu, &v("output_norm.weight")?)?,
            output,
            experts,
            slots1,
            second,
            embed: m("token_embd.weight")?,
            host_x: gpu.mapped((4 * n * 4 * batch) as u64)?,
            order: gpu.mapped((hp.layers * 2 * hp.experts * 4) as u64)?,
            usage: gpu.mapped((hp.layers * hp.experts * 4) as u64)?,
            batch,
            copied: if gpu.has_side() {
                Some(gpu.timeline()?)
            } else {
                None
            },
            consumed: if gpu.has_side() {
                Some(gpu.timeline()?)
            } else {
                None
            },
            streamed: std::sync::atomic::AtomicU64::new(0),
            stage,
            piece,
            snap,
            exact: std::sync::atomic::AtomicBool::new(false),
            mtp,
            hall: gpu.buffer((batch * n * 4) as u64)?,
            mpart: gpu.buffer((16 * hp.heads * 514 * 4) as u64)?,
            logits_t: gpu.buffer((spec::SMALL * hp.vocab * 4) as u64)?,
            vin: gpu.mapped((spec::SMALL * (2 * hp.experts_used + n) * 4) as u64)?,
            cpu_t: gpu.mapped((spec::SMALL * n * 4) as u64)?,
            snaps: drafts,
            trade_turn: std::sync::atomic::AtomicUsize::new(0),
            gslots: if drafts > 0 && !stage_empty {
                (0..hp.layers)
                    .map(|_| gpu.mapped((hp.experts * 4) as u64))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            },
            route: gpu.mapped(((2 * hp.experts_used + n + 4) * 4) as u64)?,
            cpu_out: gpu.mapped((n * 4) as u64)?,
            token: gpu.mapped((spec::SMALL * 4) as u64)?,
            s,
            blocks,
            model,
            gpu,
            k,
            e,
            pos: 0,
            max_positions,
            trace: None,
            trade_stage,
            trades: std::sync::Mutex::new(Vec::new()),
            profile: std::sync::Mutex::new(Default::default()),
            split: std::sync::Mutex::new([0.0; 2]),
        })
    }

    /// Device bank slots per MoE block.
    pub fn bank_slots(&self) -> usize {
        self.slots1
    }

    /// Runs `token` at the next position; returns the most likely next token.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        self.load_all()?;
        let hp = self.model.hp.clone();
        let n = hp.embd;
        if self.pos >= self.max_positions {
            return Err("context_full");
        }
        let mut row = vec![0f32; n];
        self.embed.decode_row(token as usize, &mut row)?;
        let x4: Vec<f32> = (0..4).flat_map(|_| row.iter().copied()).collect();
        self.gpu.write_mapped(&self.host_x, 0, &x4)?;
        if let Some(t) = &mut self.trace {
            t.clear();
        }
        // The streams: cur is the buffer holding them (None: the host's embedding).
        let mut cur: Option<usize> = None;
        let mut ops: Vec<Op> = Vec::new();
        let mut pending: Vec<Passing> = Vec::new();
        let this = &*self;
        let s = &this.s;
        let lane = Lane {
            s,
            host_x: &this.host_x,
            route: &this.route,
            cpu_out: &this.cpu_out,
        };
        let hpr = &hp;
        let mut traces = Vec::new();
        // Seconds waiting for the router, in the CPU's experts.
        let mut timing = [0f64; 3];
        let started = std::time::Instant::now();
        let mut next_mode = 0u32;
        let mut next_o: &Buffer = &s.out;
        let mut next_o3: &Buffer = &s.zeros;
        for (il, blk) in this.blocks.iter().enumerate() {
            // The attention mixer (applying the previous block's FFN mixer first).
            let c = mix(
                this, lane, hpr, &mut ops, next_o3, cur, &blk.attn, next_mode, next_o,
            );
            cur = Some(c);
            if this.trace.is_some() && il > 0 {
                // The previous block's result is now the streams.
                pending.push(submit(this, &mut ops)?);
                for p in pending.drain(..) {
                    p.wait()?;
                }
                traces.push(self_mean(&this.gpu, &s.x[c], n)?);
            }
            one_attention(this, lane, &mut ops, &blk.attention, this.pos);
            // The FFN mixer, applying the attention's (output in a).
            let c = mix(
                this,
                lane,
                hpr,
                &mut ops,
                next_o3,
                cur,
                &blk.ffn_mixer,
                1,
                &s.a,
            );
            cur = Some(c);
            match &blk.ffn {
                Ffn::Dense { gu, down } => {
                    (next_mode, next_o) = one_dense(this, lane, &mut ops, il, gu, down);
                }
                Ffn::Moe {
                    router,
                    bias,
                    gu,
                    down,
                } => {
                    one_router(this, lane, &mut ops, il, router, bias, this.pos);
                    let routed = submit(this, &mut ops)?;
                    one_back(this, lane, &mut ops, il, gu, down)?;
                    let ex = this.experts[il].get().ok_or("model_state_invalid")?;
                    pending.push(submit(this, &mut ops)?);
                    let t = std::time::Instant::now();
                    this.await_route(seq(this.pos, il), routed, &mut pending)?;
                    timing[0] += t.elapsed().as_secs_f64();
                    // Trade a recurring CPU expert onto the device while the CPU works.
                    if let Some(p) = this.trade(il, ex)? {
                        pending.push(p);
                    }
                    let t = std::time::Instant::now();
                    let sum = if std::env::var_os("KUNO_GLM_SKIP_EXPERTS").is_some() {
                        vec![0f32; n]
                    } else {
                        this.cpu_experts(il, ex)?
                    };
                    timing[1] += t.elapsed().as_secs_f64();
                    if let Some(ex) = this.experts[il].get() {
                        let mut r = vec![0f32; hp.experts_used];
                        this.gpu.read_mapped(&this.route, 0, &mut r)?;
                        timing[2] += r
                            .iter()
                            .filter(|v| {
                                let e = v.to_bits() as usize;
                                ex.slot[e].is_none() && ex.slot2[e].is_none()
                            })
                            .count() as f64;
                    }
                    this.gpu.write_mapped(&this.cpu_out, 0, &sum)?;
                    next_mode = 2;
                    next_o3 = if ex.device.is_some() { &s.gd } else { &s.zeros };
                    next_o = &s.a;
                }
            }
        }
        // The last block's FFN mixer (with any block's product, unused), the norm, the head.
        let last = &this.blocks[0].attn;
        let c = mix(
            this, lane, hpr, &mut ops, next_o3, cur, last, next_mode, next_o,
        );
        if this.trace.is_some() {
            pending.push(submit(this, &mut ops)?);
            for p in pending.drain(..) {
                p.wait()?;
            }
            traces.push(self_mean(&this.gpu, &s.x[c], n)?);
        }
        ops.push(Op {
            kernel: &this.e.fin,
            buffers: vec![&s.x[c], &this.out_norm, &s.normed],
            constants: vec![n as u32, f(hp.rms_eps)],
            groups: [1, 1, 1],
            independent: false,
        });
        mv(
            this,
            &mut ops,
            &this.output,
            &s.normed,
            0,
            &s.logits,
            0,
            this.output.constants(0, 0, false),
            false,
        );
        ops.push(Op {
            kernel: &this.e.argmax,
            buffers: vec![&s.logits, this.token.buffer()],
            constants: vec![hp.vocab as u32, 0],
            groups: [1, 1, 1],
            independent: false,
        });
        let last = submit(this, &mut ops)?;
        for p in pending.drain(..) {
            p.wait()?;
        }
        last.wait()?;
        let mut t = [0f32; 1];
        this.gpu.read_mapped(&this.token, 0, &mut t)?;
        drop(ops);
        self.settle_trades()?;
        if let Some(tr) = &mut self.trace {
            *tr = traces;
        }
        let split = self
            .split
            .lock()
            .map(|mut p| std::mem::take(&mut *p))
            .unwrap_or_default();
        if std::env::var_os("KUNO_GLM_TIMING").is_some() {
            eprintln!(
                "token: {:.1} ms; router waits {:.1} ms, CPU experts {:.1} ms ({} experts, {:.1} GB/s: sums {:.1} ms, second device waits {:.1} ms)",
                started.elapsed().as_secs_f64() * 1e3,
                timing[0] * 1e3,
                timing[1] * 1e3,
                timing[2],
                timing[2] * 8.97e6 / timing[1] / 1e9,
                split[0] * 1e3,
                split[1] * 1e3
            );
        }
        self.pos += 1;
        Ok(t[0].to_bits())
    }

    /// A reader for loading experts.
    fn reader(&self) -> Result<experts::Reader<'m>> {
        self.model.reader()
    }

    /// Loads every block's experts not loaded yet (the bank by recent use where a prompt
    /// counted it, else in index order). Callers may run it while idle.
    pub fn load_all(&self) -> Result<()> {
        let hp = &self.model.hp;
        if (hp.dense_lead..hp.layers).all(|il| self.experts[il].get().is_some()) {
            return Ok(());
        }
        // Each block read by threads of its own while the one before is placed.
        let todo: Vec<usize> = (hp.dense_lead..hp.layers)
            .filter(|&il| self.experts[il].get().is_none())
            .collect();
        let model = self.model;
        std::thread::scope(|sc| -> Result<()> {
            let first = todo[0];
            let host = self.model.hp.experts - self.slots1;
            let gpu = &self.gpu;
            let mut next = Some(sc.spawn(move || experts::prefetch_block(model, gpu, first, host)));
            for (i, &il) in todo.iter().enumerate() {
                let (raws, arena) = next
                    .take()
                    .ok_or("model_read_failed")?
                    .join()
                    .map_err(|_| "model_read_failed")??;
                if let Some(&j) = todo.get(i + 1) {
                    next = Some(sc.spawn(move || experts::prefetch_block(model, gpu, j, host)));
                }
                self.load_block(il, None, Some(raws), Some(arena))?;
            }
            Ok(())
        })
    }

    /// Places block `il`'s experts from `raws` (every expert's raw rows): the bank takes
    /// the experts `counts` chose most (ties and the rest in index order).
    fn load_block(
        &self,
        il: usize,
        counts: Option<&[u32]>,
        raws: Option<Vec<Vec<u8>>>,
        arena: Option<Vec<Mapped>>,
    ) -> Result<()> {
        let e = self.model.hp.experts;
        let mut ids: Vec<usize> = (0..e).collect();
        if let Some(c) = counts {
            ids.sort_by(|a, b| c[*b].cmp(&c[*a]).then(a.cmp(b)));
        }
        let bank = &ids[..self.slots1.min(e)];
        let reader = self.reader()?;
        let got = |x: usize| -> Result<std::borrow::Cow<'_, [u8]>> {
            match &raws {
                Some(r) => Ok(std::borrow::Cow::Borrowed(&r[x][..])),
                None => {
                    experts::read_expert(self.model, &reader, il, x).map(std::borrow::Cow::Owned)
                }
            }
        };
        let block = experts::place_block(self.model, &self.gpu, il, bank, None, &got, arena)?;
        let _ = self.experts[il].set(block);
        self.write_order_block(il)
    }

    /// Plans block `il`'s trade for this token's choice and submits its copies (victim down
    /// to staging, then the expert up into its slot and the new slot map).
    fn trade<'a>(&'a self, il: usize, ex: &'a BlockExperts) -> Result<Option<Passing>> {
        let hp = &self.model.hp;
        let (Some(_), Ok(mut trades)) = (&ex.device, self.trades.lock()) else {
            return Ok(None);
        };
        if trades.len() >= self.trade_stage.len() {
            // The budget is spent: still count the choice.
            let mut r = vec![0f32; hp.experts_used];
            self.gpu.read_mapped(&self.route, 0, &mut r)?;
            let ids: Vec<usize> = r.iter().map(|v| v.to_bits() as usize).collect();
            let _ = ex.observe(il, &ids, self.pos as u64, f32::INFINITY);
            return Ok(None);
        }
        let mut r = vec![0f32; hp.experts_used];
        self.gpu.read_mapped(&self.route, 0, &mut r)?;
        let ids: Vec<usize> = r.iter().map(|v| v.to_bits() as usize).collect();
        let margin = std::env::var("KUNO_GLM_TRADE_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.0f32);
        let Some(trade) = ex.observe(il, &ids, self.pos as u64, margin) else {
            return Ok(None);
        };
        let map = &self.trade_stage[trades.len()];
        let mut ops: Vec<Op<'a>> = Vec::new();
        self.trade_ops(ex, &trade, map, &mut ops)?;
        trades.push(trade);
        drop(trades);
        // Runs after this block's device experts (queue order), while the CPU works.
        Ok(Some(submit(self, &mut ops)?))
    }

    /// Records `trade`'s copies (victim down to its spare host slot, then the expert up into
    /// its slot and the new slot map staged in `map`).
    fn trade_ops<'a>(
        &'a self,
        ex: &'a BlockExperts,
        trade: &experts::Trade,
        map: &'a Mapped,
        ops: &mut Vec<Op<'a>>,
    ) -> Result<()> {
        let hp = &self.model.hp;
        let pool = ex.device.as_ref().ok_or("model_state_invalid")?;
        let slots = ex.slots_after(trade);
        self.gpu
            .write_mapped(map, 0, &kuno_image::engine::gpu::floats(bytes_of(&slots)))?;
        let pools = [&pool.gu_w, &pool.gu_s, &pool.down_w, &pool.down_s];
        let copy = |ops: &mut Vec<Op<'a>>,
                    src: &'a Buffer,
                    dst: &'a Buffer,
                    n: usize,
                    so: usize,
                    d_o: usize,
                    indep: bool| {
            let groups = (n as u32).div_ceil(256);
            ops.push(Op {
                kernel: &self.e.copyoff,
                buffers: vec![src, dst],
                constants: vec![n as u32, so as u32, d_o as u32],
                groups: [groups.min(65535), groups.div_ceil(65535), 1],
                independent: indep,
            });
        };
        // The victim out to the spare host slot, then the expert in from its host slot.
        if trade.down.is_some() {
            let (chunk, mut at) = ex.host_at(trade.spare);
            for (b, &st) in pools.iter().zip(&pool.stride) {
                if st > 0 {
                    copy(ops, b, chunk.buffer(), st, trade.slot * st, at, true);
                }
                at += st;
            }
            if let Some(last) = ops.last_mut() {
                last.independent = false;
            }
        }
        let (chunk, mut at) = ex.host_at(ex.host_slot[trade.up].ok_or("model_state_invalid")?);
        for (b, &st) in pools.iter().zip(&pool.stride) {
            if st > 0 {
                copy(ops, chunk.buffer(), b, st, at, trade.slot * st, true);
            }
            at += st;
        }
        copy(ops, map.buffer(), &pool.slots, hp.experts, 0, 0, false);
        Ok(())
    }

    /// After the token: every trade's host side (its passes have completed).
    fn settle_trades(&mut self) -> Result<()> {
        let trades = std::mem::take(&mut *self.trades.lock().map_err(|_| "gpu_failed")?);
        for t in &trades {
            self.experts[t.block]
                .get_mut()
                .ok_or("model_state_invalid")?
                .settle(t);
        }
        if std::env::var_os("KUNO_GLM_TIMING").is_some() && !trades.is_empty() {
            eprintln!("traded {} experts", trades.len());
        }
        Ok(())
    }

    /// Waits until the router's choice `seq` is visible in `route`: spinning on the word
    /// route.comp writes last (cheaper than the fence's wake-up), the fence after 2 s.
    fn await_route(&self, seq: u32, routed: Passing, pending: &mut Vec<Passing>) -> Result<()> {
        let at = ((2 * self.model.hp.experts_used + self.model.hp.embd) * 4) as u64;
        let started = std::time::Instant::now();
        let mut word = [0f32; 1];
        loop {
            self.gpu.read_mapped(&self.route, at, &mut word)?;
            if word[0].to_bits() == seq {
                pending.push(routed);
                return Ok(());
            }
            if started.elapsed().as_secs() >= 2 {
                routed.wait()?;
                self.gpu.read_mapped(&self.route, at, &mut word)?;
                return if word[0].to_bits() == seq {
                    Ok(())
                } else {
                    Err("gpu_failed")
                };
            }
            std::hint::spin_loop();
        }
    }

    /// The routed experts of block `il` kept in memory, on the CPU, for the router's
    /// choice in `route`: their weighted sum (the device adds its own).
    fn cpu_experts(&self, il: usize, ex: &BlockExperts) -> Result<Vec<f32>> {
        let hp = &self.model.hp;
        let (n, used) = (hp.embd, hp.experts_used);
        let mut r = vec![0f32; 2 * used + n];
        self.gpu.read_mapped(&self.route, 0, &mut r)?;
        let ids: Vec<usize> = r[..used].iter().map(|v| v.to_bits() as usize).collect();
        // The second device's experts run while the CPU runs its own.
        let second = match (&self.second, &ex.device2) {
            (Some(sd), Some(pool)) if ids.iter().any(|e| ex.slot2[*e].is_some()) => {
                Some(sd.begin(pool, ex.gate_kind, ex.down_kind, &r, hp.clamp_expert[il])?)
            }
            _ => None,
        };
        let t = std::time::Instant::now();
        let mut sum = ex.cpu_sum(
            &ids,
            &r[used..2 * used],
            &r[2 * used..],
            hp.ff_expert,
            hp.clamp_expert[il],
        );
        let cpu = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        if let (Some(passing), Some(sd)) = (second, &self.second) {
            sd.finish(passing, &mut sum)?;
        }
        if let Ok(mut p) = self.split.lock() {
            p[0] += cpu;
            p[1] += t.elapsed().as_secs_f64();
        }
        Ok(sum)
    }
}

/// One token's buffers: its scratch, its embedding (x4), the router's choice for the host
/// and the CPU's expert sum (`step`'s own, or a lane of a pipelined verification).
#[derive(Clone, Copy)]
struct Lane<'a> {
    s: &'a Scratch,
    host_x: &'a Mapped,
    route: &'a Mapped,
    cpu_out: &'a Mapped,
}

/// A lane's own buffers (one token's scratch).
struct LaneBufs {
    s: Scratch,
    host_x: Mapped,
    route: Mapped,
    cpu_out: Mapped,
}

impl Scratch {
    /// Scratch for `batch` tokens; `lp` the expert products' rows (assignments).
    fn new(
        gpu: &Gpu,
        hp: &crate::Hparams,
        batch: usize,
        lp: usize,
        max_positions: usize,
    ) -> Result<Self> {
        let n = hp.embd;
        let buf = |floats: usize| gpu.buffer((floats * batch * 4) as u64);
        let used = hp.experts_used;
        Ok(Scratch {
            x: [buf(4 * n)?, buf(4 * n)?],
            part: buf(32 * 32)?,
            coef: buf(32)?,
            normed: buf(n)?,
            a: buf(3 * hp.kda_head * hp.heads)?,
            b: buf(hp.heads * hp.kv_lora.max(2 * hp.kda_head))?,
            c: buf(hp.heads * hp.kv_lora)?,
            out: buf(hp.ff.max(2 * hp.ff_shared) * 2)?,
            logits: gpu.buffer((hp.vocab * 4) as u64)?,
            // A batch's assignments by expert, each expert's run padded to 8 (group.comp).
            ge: gpu.buffer((lp * 2 * hp.ff_expert * 4) as u64)?,
            gh: gpu.buffer((lp * hp.ff_expert * 4) as u64)?,
            gd: gpu.buffer((lp * n * 4) as u64)?,
            zeros: floats(gpu, &vec![0f32; n])?,
            routed: buf(2 * used)?,
            ia: buf(256 + 32)?,
            ib: buf(32 * 128)?,
            iscore: buf((max_positions / 4).max(4))?,
            iidx: buf(hp.indexer_top_k + hp.indexer_kpool)?,
            lists: gpu.buffer(((2 * hp.experts + 2 * lp) * 4) as u64)?,
        })
    }
}

impl LaneBufs {
    /// One token's lane (a pipelined verification's).
    fn new(gpu: &Gpu, hp: &crate::Hparams, max_positions: usize) -> Result<Self> {
        let n = hp.embd;
        Ok(Self {
            s: Scratch::new(gpu, hp, 1, hp.experts_used, max_positions)?,
            host_x: gpu.mapped((4 * n * 4) as u64)?,
            route: gpu.mapped(((2 * hp.experts_used + n + 4) * 4) as u64)?,
            cpu_out: gpu.mapped((n * 4) as u64)?,
        })
    }
}

impl LaneBufs {
    fn lane(&self) -> Lane<'_> {
        Lane {
            s: &self.s,
            host_x: &self.host_x,
            route: &self.route,
            cpu_out: &self.cpu_out,
        }
    }
}

/// One token's attention (at `pos`) from `s.normed`, its output in `s.a`.
fn one_attention<'a>(
    this: &'a Engine,
    lane: Lane<'a>,
    ops: &mut Vec<Op<'a>>,
    attention: &'a Attention,
    pos: usize,
) {
    let hp = &this.model.hp;
    let s = lane.s;
    match attention {
        Attention::Kda {
            qkv,
            small,
            fg_b,
            conv,
            params,
            out,
            hist,
            state,
        } => {
            let d = hp.kda_head * hp.heads;
            mv(
                this,
                ops,
                qkv,
                &s.normed,
                0,
                &s.a,
                0,
                qkv.constants(0, 0, false),
                true,
            );
            mv(
                this,
                ops,
                small,
                &s.normed,
                0,
                &s.c,
                0,
                small.constants(0, 0, false),
                false,
            );
            // f_b on f_a's 128, g_b on g_a's 128 (one product, two input groups):
            // fg [f | gate] in b.
            mv(
                this,
                ops,
                fg_b,
                &s.c,
                0,
                &s.b,
                0,
                fg_b.grouped(0, 0, d, 128),
                false,
            );
            ops.push(Op {
                kernel: &this.e.kda,
                buffers: vec![&s.a, conv, hist, &s.b, &s.c, params, state, &s.out],
                constants: vec![d as u32, f(hp.gate_lower_bound), f(hp.rms_eps), 256],
                groups: [hp.heads as u32, 1, 1],
                independent: false,
            });
            mv(
                this,
                ops,
                out,
                &s.out,
                0,
                &s.a,
                0,
                out.constants(0, 0, false),
                false,
            );
        }
        Attention::Dsa {
            q_a,
            kv_a,
            norms,
            q_b,
            k_b,
            v_b,
            out,
            latents,
            idx_kg,
            idx_proj,
            idx_qb,
            idx_params,
            idx_keys,
            idx_pooled,
        } => {
            let (ql, lora) = (hp.q_lora, hp.kv_lora);
            // The indexer's key|gate and head weights, from the block's input.
            mv(
                this,
                ops,
                idx_kg,
                &s.normed,
                0,
                &s.ia,
                0,
                idx_kg.constants(0, 0, false),
                true,
            );
            let mut pc = idx_proj.constants(0, 256, false);
            pc[6] |= 4;
            mv(this, ops, idx_proj, &s.normed, 0, &s.ia, 256, pc, true);
            mv(
                this,
                ops,
                q_a,
                &s.normed,
                0,
                &s.c,
                0,
                q_a.constants(0, 0, false),
                true,
            );
            mv(
                this,
                ops,
                kv_a,
                &s.normed,
                0,
                &s.c,
                ql,
                kv_a.constants(0, ql, false),
                false,
            );
            ops.push(Op {
                kernel: &this.e.dsanorm,
                buffers: vec![&s.c, norms, &s.out, latents],
                constants: vec![ql as u32, lora as u32, pos as u32, f(hp.rms_eps)],
                groups: [1, 1, 1],
                independent: false,
            });
            ops.push(Op {
                kernel: &this.e.idxstore,
                buffers: vec![&s.ia, idx_params, idx_keys, idx_pooled],
                constants: vec![pos as u32, f(1e-6)],
                groups: [1, 1, 1],
                independent: false,
            });
            // Past top_k + kpool - 1 positions the indexer chooses the pools attended.
            let kpool = hp.indexer_kpool;
            let pools = (pos + 1) / kpool;
            let keep = hp.indexer_top_k / kpool;
            let indexed = pools > keep;
            let ntail = pos + 1 - pools * kpool;
            if indexed {
                mv(
                    this,
                    ops,
                    idx_qb,
                    &s.out,
                    0,
                    &s.ib,
                    0,
                    idx_qb.constants(0, 0, false),
                    false,
                );
                ops.push(Op {
                    kernel: &this.e.idxscore,
                    buffers: vec![&s.ib, &s.ia, idx_pooled, &s.iscore],
                    constants: vec![pools as u32, f(1.0 / ((128 * 32) as f32).sqrt()), 256],
                    groups: [(pools as u32).div_ceil(128), 1, 1],
                    independent: false,
                });
                ops.push(Op {
                    kernel: &this.e.idxselect,
                    buffers: vec![&s.iscore, &s.iidx],
                    constants: vec![
                        pools as u32,
                        keep as u32,
                        (pools * kpool) as u32,
                        ntail as u32,
                    ],
                    groups: [1, 1, 1],
                    independent: false,
                });
            }
            let count = if indexed {
                keep * kpool + ntail
            } else {
                pos + 1
            };
            // q [heads][qk] in a; absorbed [heads][lora] in b; mixed in c; heads' values in out.
            mv(
                this,
                ops,
                q_b,
                &s.out,
                0,
                &s.a,
                0,
                q_b.constants(0, 0, false),
                false,
            );
            mv(
                this,
                ops,
                k_b,
                &s.a,
                0,
                &s.b,
                0,
                k_b.grouped(0, 0, lora, hp.qk_head),
                false,
            );
            ops.push(Op {
                kernel: &this.e.mla,
                buffers: vec![&s.b, latents, &s.c, &s.iidx],
                constants: vec![
                    lora as u32,
                    count as u32,
                    f(1.0 / (hp.qk_head as f32).sqrt()),
                    indexed as u32,
                ],
                groups: [hp.heads as u32, 1, 1],
                independent: false,
            });
            mv(
                this,
                ops,
                v_b,
                &s.c,
                0,
                &s.out,
                0,
                v_b.grouped(0, 0, hp.v_head, lora),
                false,
            );
            mv(
                this,
                ops,
                out,
                &s.out,
                0,
                &s.a,
                0,
                out.constants(0, 0, false),
                false,
            );
        }
    }
}

/// One token's dense FFN from `s.normed`: the next mixer's mode and output.
fn one_dense<'a>(
    this: &'a Engine,
    lane: Lane<'a>,
    ops: &mut Vec<Op<'a>>,
    il: usize,
    gu: &'a Mat,
    down: &'a Mat,
) -> (u32, &'a Buffer) {
    let hp = &this.model.hp;
    let s = lane.s;
    let ff = gu.rows / 2;
    mv(
        this,
        ops,
        gu,
        &s.normed,
        0,
        &s.out,
        0,
        gu.constants(0, 0, false),
        false,
    );
    ops.push(Op {
        kernel: &this.e.glu,
        buffers: vec![&s.out, &s.b],
        constants: vec![ff as u32, f(hp.clamp_shared[il]), 1],
        groups: [(ff as u32).div_ceil(256), 1, 1],
        independent: false,
    });
    mv(
        this,
        ops,
        down,
        &s.b,
        0,
        &s.a,
        0,
        down.constants(0, 0, false),
        false,
    );
    (1, &s.a)
}

/// One token's router (at `pos`) into `lane.route` (its seq last) and `s.routed`.
#[allow(clippy::too_many_arguments)]
fn one_router<'a>(
    this: &'a Engine,
    lane: Lane<'a>,
    ops: &mut Vec<Op<'a>>,
    il: usize,
    router: &'a Mat,
    bias: &'a Buffer,
    pos: usize,
) {
    let hp = &this.model.hp;
    let s = lane.s;
    let n = hp.embd;
    mv(
        this,
        ops,
        router,
        &s.normed,
        0,
        &s.c,
        0,
        {
            // Few rows: a whole workgroup per row.
            let mut c = router.constants(0, 0, false);
            c[6] |= 4;
            c
        },
        false,
    );
    ops.push(Op {
        kernel: &this.e.route,
        buffers: vec![&s.c, bias, &s.normed, lane.route.buffer(), &s.routed],
        constants: vec![
            hp.experts as u32,
            hp.experts_used as u32,
            f(hp.expert_scale),
            n as u32,
            seq(pos, il),
        ],
        groups: [1, 1, 1],
        independent: false,
    });
}

/// One token's shared expert (into `s.a`) and the device's bank of block `il` (into
/// `s.gd`, returned; zeros without a bank).
fn one_back<'a>(
    this: &'a Engine,
    lane: Lane<'a>,
    ops: &mut Vec<Op<'a>>,
    il: usize,
    gu: &'a Mat,
    down: &'a Mat,
) -> Result<&'a Buffer> {
    let hp = &this.model.hp;
    let s = lane.s;
    let n = hp.embd;
    // The shared expert, queued behind the router while the CPU works.
    let ff = gu.rows / 2;
    mv(
        this,
        ops,
        gu,
        &s.normed,
        0,
        &s.out,
        0,
        gu.constants(0, 0, false),
        false,
    );
    ops.push(Op {
        kernel: &this.e.glu,
        buffers: vec![&s.out, &s.b],
        constants: vec![ff as u32, f(hp.clamp_shared[il]), 1],
        groups: [(ff as u32).div_ceil(256), 1, 1],
        independent: false,
    });
    mv(
        this,
        ops,
        down,
        &s.b,
        0,
        &s.a,
        0,
        down.constants(0, 0, false),
        false,
    );
    let ex = this.experts[il].get().ok_or("model_state_invalid")?;
    if let Some(pool) = &ex.device {
        let (ff, used) = (hp.ff_expert, hp.experts_used);
        let rpg = 128 / this.k.lanes;
        let gk = experts::kind(ex.gate_kind)?;
        let dk = experts::kind(ex.down_kind)?;
        ops.push(Op {
            kernel: &this.e.mve,
            buffers: vec![
                &pool.gu_w,
                &pool.gu_s,
                &s.normed,
                &s.ge,
                &s.routed,
                &pool.slots,
                &this.e.tables,
            ],
            constants: vec![0, gk, (2 * ff) as u32, n as u32, used as u32, 0, 0, 0, 0, 0],
            groups: [((2 * ff) as u32).div_ceil(rpg), used as u32, 1],
            independent: false,
        });
        ops.push(Op {
            kernel: &this.e.glu,
            buffers: vec![&s.ge, &s.gh],
            constants: vec![ff as u32, f(hp.clamp_expert[il]), used as u32],
            groups: [((used * ff) as u32).div_ceil(256), 1, 1],
            independent: false,
        });
        ops.push(Op {
            kernel: &this.e.mve,
            buffers: vec![
                &pool.down_w,
                &pool.down_s,
                &s.gh,
                &s.gd,
                &s.routed,
                &pool.slots,
                &this.e.tables,
            ],
            constants: vec![1, dk, n as u32, ff as u32, used as u32, 0, 0, 0, 0, 0],
            groups: [(n as u32).div_ceil(rpg), 1, 1],
            independent: false,
        });
    }
    Ok(if ex.device.is_some() { &s.gd } else { &s.zeros })
}

/// Records `m x` (constants `c`) into `y`.
#[allow(clippy::too_many_arguments)]
fn mv<'a>(
    this: &'a Engine,
    ops: &mut Vec<Op<'a>>,
    m: &'a Mat,
    x: &'a Buffer,
    _x_off: usize,
    y: &'a Buffer,
    _y_off: usize,
    c: [u32; 9],
    independent: bool,
) {
    let groups = if c[6] & 4 != 0 {
        c[0]
    } else {
        c[0].div_ceil(128 / this.k.lanes)
    };
    ops.push(Op {
        kernel: &this.k.mv,
        buffers: vec![&m.w, &m.s, x, y],
        constants: c.to_vec(),
        groups: [groups.min(65535), groups.div_ceil(65535), 1],
        independent,
    });
}

/// hcdot (mode 0 on cur; 1 or 2 applying the previous mixer, the output `o`, into the other
/// buffer; 3 copying the host's embedding) then hcmix with `mixer`; returns the new cur.
#[allow(clippy::too_many_arguments)]
fn mix<'a>(
    this: &'a Engine,
    lane: Lane<'a>,
    hp: &crate::Hparams,
    ops: &mut Vec<Op<'a>>,
    o3: &'a Buffer,
    cur: Option<usize>,
    mixer: &'a Mixer,
    mode: u32,
    o: &'a Buffer,
) -> usize {
    let s = lane.s;
    let n = hp.embd;
    let (src, dst) = match cur {
        None => (lane.host_x.buffer(), 0),
        Some(i) => (&s.x[i], 1 - i),
    };
    let mode = if cur.is_none() { 3 } else { mode };
    let target = if mode == 0 { cur.unwrap_or(0) } else { dst };
    let xn = &s.x[dst];
    ops.push(Op {
        kernel: &this.e.hcdot,
        buffers: vec![
            &mixer.function.w,
            &mixer.function.s,
            src,
            &s.part,
            &s.coef,
            o,
            xn,
            lane.cpu_out.buffer(),
            o3,
        ],
        constants: vec![n as u32, mode, 0],
        groups: [32, 1, 1],
        independent: false,
    });
    let xcur: &Buffer = if mode == 0 { src } else { xn };
    ops.push(Op {
        kernel: &this.e.hcmix,
        buffers: vec![
            &s.part,
            xcur,
            &mixer.params,
            &mixer.norm,
            &s.coef,
            &s.normed,
        ],
        constants: vec![
            n as u32,
            hp.sinkhorn as u32,
            f(hp.rms_eps),
            f(hp.hc_eps),
            32,
        ],
        groups: [1, 1, 1],
        independent: false,
    });
    target
}

/// Submits the recorded steps.
fn submit<'a>(this: &'a Engine, ops: &mut Vec<Op<'a>>) -> Result<Passing> {
    let steps: Vec<Step> = ops
        .iter()
        .map(|o| Step {
            kernel: o.kernel,
            buffers: &o.buffers,
            constants: &o.constants,
            groups: o.groups,
        })
        .collect();
    let independent: Vec<bool> = ops.iter().map(|o| o.independent).collect();
    if std::env::var_os("KUNO_GLM_PROFILE").is_some() && !steps.is_empty() {
        // Each step timed (with barriers after every step: device time, not overlap).
        let times = this.gpu.pass_profiled(&steps, &vec![false; steps.len()])?;
        let e = &this.e;
        let names: &[(&Kernel, &str)] = &[
            (&e.b.hcdot, "hcdot_b"),
            (&e.b.hcmix, "hcmix_b"),
            (&e.b.kda, "kda_b"),
            (&e.b.dsanorm, "dsanorm_b"),
            (&e.b.idxstore, "idxstore_b"),
            (&e.b.idxscore, "idxscore_b"),
            (&e.b.idxselect, "idxselect_b"),
            (&e.b.mla, "mla_b"),
            (&e.b.route, "route_b"),
            (&e.b.mm, "mm"),
            (&e.b.mvt, "mvt"),
            (&e.b.mvr, "mvr"),
            (&e.b.mvr2, "mvr2"),
            (e.b.mmc.as_ref().unwrap_or(&e.b.mm), "mmc"),
            (e.b.mec.as_ref().unwrap_or(&e.b.mm), "mec"),
            (&e.b.group, "group"),
            (&e.b.combine, "combine"),
            (&e.b.copy4, "copy4"),
            (&e.copyoff, "copyoff"),
            (&e.hcdot, "hcdot"),
            (&e.hcmix, "hcmix"),
            (&e.kda, "kda"),
            (&e.dsanorm, "dsanorm"),
            (&e.mla, "mla"),
            (&e.route, "route"),
            (&e.glu, "glu"),
            (&e.fin, "final"),
            (&e.argmax, "argmax"),
            (&e.mve, "mve"),
            (&this.k.mv, "mv"),
        ];
        let mut p = this.profile.lock().map_err(|_| "gpu_failed")?;
        for (o, t) in ops.iter().zip(&times) {
            let name = names
                .iter()
                .find(|(k, _)| std::ptr::eq(*k, o.kernel))
                .map_or("?", |n| n.1);
            let label = match name {
                "mv" => format!(
                    "mv {}x{} kind {}",
                    o.constants[0], o.constants[1], o.constants[2]
                ),
                "mve" => format!("mve mode {} kind {}", o.constants[0], o.constants[1]),
                "mm" | "mvt" | "mvr" | "mvr2" => format!(
                    "{name} {}x{} kind {} x{}",
                    o.constants[0], o.constants[1], o.constants[2], o.constants[11]
                ),
                "mmc" => format!(
                    "mmc {}x{} kind {} x{}",
                    o.constants[0], o.constants[1], o.constants[2], o.constants[9]
                ),
                "mec" => format!(
                    "mec mode {} kind {} slots {}",
                    o.constants[0], o.constants[1], o.groups[1]
                ),
                n => n.to_string(),
            };
            let entry = p.entry(label).or_insert((0.0, 0));
            entry.0 += t;
            entry.1 += 1;
        }
    }
    // SAFETY: every buffer belongs to the engine, which outlives the pass (waited for before
    // `step` returns).
    let passing = unsafe { this.gpu.pass_begin(&steps, &independent)? };
    ops.clear();
    Ok(passing)
}

fn self_mean(gpu: &Gpu, x: &Buffer, n: usize) -> Result<Vec<f32>> {
    self_mean_at(gpu, x, n, 0)
}

/// The mean of token `token`'s four streams.
fn self_mean_at(gpu: &Gpu, x: &Buffer, n: usize, token: usize) -> Result<Vec<f32>> {
    let stage = gpu.readback((4 * n * 4) as u64)?;
    let mut v = vec![0f32; 4 * n];
    gpu.read_floats(x, (token * 4 * n * 4) as u64, &stage, &mut v)?;
    Ok((0..n)
        .map(|i| (0..4).map(|h| v[h * n + i]).sum::<f32>() / 4.0)
        .collect())
}
