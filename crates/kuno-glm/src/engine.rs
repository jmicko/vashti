//! Decoding one token at a time on the GPU, the routed experts on the CPU: each MoE block's
//! work up to the router is one submission; the host reads the chosen experts, runs them
//! (`cpu`), and the next submission adds their sum while it applies the block's mixer.
use kuno_image::engine::gpu::{Buffer, Gpu, Kernel, Mapped, Passing, Step, bytes_of};

use crate::experts::{self, BlockExperts};
use crate::gpu::{Kernels, Mat, filled};
use crate::{Matrix, Model, Result};

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
    /// The IQ2_S and IQ3_S grids for mve.
    tables: Buffer,
}

struct Mixer {
    function: Mat,
    /// base [24], scale [3]
    params: Buffer,
    norm: Buffer,
}

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

/// Device scratch for one token.
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
}

pub struct Engine<'m> {
    model: &'m Model,
    gpu: Gpu,
    k: Kernels,
    e: Extra,
    blocks: Vec<Block>,
    experts: Vec<Option<BlockExperts>>,
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
    /// When set, the mean of the streams after each block (comparison with the reference).
    pub trace: Option<Vec<Vec<f32>>>,
    /// How often each block's experts were chosen ([block][expert]; placement statistics).
    pub counts: std::sync::Mutex<Vec<Vec<u32>>>,
    /// With KUNO_GLM_PROFILE: device time per kernel (and product shape), ns and count.
    pub profile: std::sync::Mutex<std::collections::BTreeMap<String, (f64, u32)>>,
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
        let k = Kernels::new(&gpu)?;
        let lanes = Some(k.lanes);
        let kernel = |spirv: &[u8], b, c| gpu.kernel(spirv, b, c, lanes);
        let e = Extra {
            hcdot: kernel(include_bytes!("../shaders/hcdot.spv"), 9, 3)?,
            hcmix: kernel(include_bytes!("../shaders/hcmix.spv"), 6, 5)?,
            kda: kernel(include_bytes!("../shaders/kda.spv"), 8, 4)?,
            dsanorm: kernel(include_bytes!("../shaders/dsanorm.spv"), 4, 4)?,
            mla: kernel(include_bytes!("../shaders/mla.spv"), 3, 3)?,
            route: kernel(include_bytes!("../shaders/route.spv"), 5, 5)?,
            glu: kernel(include_bytes!("../shaders/glu.spv"), 2, 3)?,
            fin: kernel(include_bytes!("../shaders/final.spv"), 3, 2)?,
            argmax: gpu.kernel(include_bytes!("../shaders/argmax.spv"), 2, 2, None)?,
            mve: kernel(include_bytes!("../shaders/mve.spv"), 7, 6)?,
            tables: {
                let mut t: Vec<u32> = Vec::with_capacity(2560);
                for g in kuno_text::quant::IQ2_S_GRID {
                    t.push(g as u32);
                    t.push((g >> 32) as u32);
                }
                t.extend(kuno_text::quant::IQ3_S_GRID);
                filled(&gpu, bytes_of(&t))?
            },
        };
        let m = |n: &str| model.matrix(n);
        let v = |n: &str| model.vector(n);
        let up = |mx: &Matrix| Mat::upload(&gpu, mx);
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
                let mut norms = v(&b("attn_q_a_norm.weight"))?;
                norms.extend(v(&b("attn_kv_a_norm.weight"))?);
                Attention::Dsa {
                    q_a: up(&m(&b("attn_q_a.weight"))?)?,
                    kv_a: up(&m(&b("attn_kv_a_mqa.weight"))?)?,
                    norms: floats(&gpu, &norms)?,
                    q_b: up(&m(&b("attn_q_b.weight"))?)?,
                    k_b: up(&m(&b("attn_k_b.weight"))?)?,
                    v_b: up(&m(&b("attn_v_b.weight"))?)?,
                    out: up(&m(&b("attn_output.weight"))?)?,
                    latents: gpu.buffer((max_positions * hp.kv_lora * 4) as u64)?,
                }
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
            eprintln!("block {il} on the device");
        }
        let n = hp.embd;
        let buf = |floats: usize| gpu.buffer((floats * 4) as u64);
        let s = Scratch {
            x: [buf(4 * n)?, buf(4 * n)?],
            part: buf(32 * 32)?,
            coef: buf(32)?,
            normed: buf(n)?,
            a: buf(3 * hp.kda_head * hp.heads)?,
            b: buf(hp.heads * hp.kv_lora.max(2 * hp.kda_head))?,
            c: buf(hp.heads * hp.kv_lora)?,
            out: buf(hp.ff.max(2 * hp.ff_shared) * 2)?,
            logits: buf(hp.vocab)?,
            ge: buf(hp.experts_used * 2 * hp.ff_expert)?,
            gh: buf(hp.experts_used * hp.ff_expert)?,
            gd: buf(n)?,
            zeros: floats(&gpu, &vec![0f32; n])?,
            routed: buf(2 * hp.experts_used)?,
        };
        let output = up(&m("output.weight")?)?;
        // The experts: the device's free memory less a margin (or KUNO_GLM_DEVICE_GB).
        let margin = 1u64 << 30;
        let budget = std::env::var("KUNO_GLM_DEVICE_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|g| (g * 1e9) as u64)
            .unwrap_or_else(|| gpu.free_memory().unwrap_or(0).saturating_sub(margin));
        let counts =
            std::env::var_os("KUNO_GLM_COUNTS_IN").and_then(|p| experts::read_counts(p.as_ref()));
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
        let experts = experts::place(
            model,
            &gpu,
            budget,
            second.as_ref().map(|s| (&s.gpu, budget2.unwrap_or(0))),
            counts.as_deref(),
        )?;
        Ok(Self {
            out_norm: floats(&gpu, &v("output_norm.weight")?)?,
            output,
            experts,
            second,
            embed: m("token_embd.weight")?,
            host_x: gpu.mapped((4 * n * 4) as u64)?,
            route: gpu.mapped(((2 * hp.experts_used + n + 4) * 4) as u64)?,
            cpu_out: gpu.mapped((n * 4) as u64)?,
            token: gpu.mapped(16)?,
            s,
            blocks,
            model,
            gpu,
            k,
            e,
            pos: 0,
            max_positions,
            trace: None,
            counts: std::sync::Mutex::new(vec![vec![0; hp.experts]; hp.layers]),
            profile: std::sync::Mutex::new(Default::default()),
            split: std::sync::Mutex::new([0.0; 2]),
        })
    }

    /// Runs `token` at the next position; returns the most likely next token.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        let hp = self.model.hp.clone();
        let n = hp.embd;
        if self.pos >= self.max_positions || self.pos >= hp.indexer_top_k + hp.indexer_kpool - 1 {
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
                this, hpr, &mut ops, next_o3, cur, &blk.attn, next_mode, next_o,
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
            match &blk.attention {
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
                        &mut ops,
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
                        &mut ops,
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
                        &mut ops,
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
                        &mut ops,
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
                } => {
                    let (ql, lora) = (hp.q_lora, hp.kv_lora);
                    mv(
                        this,
                        &mut ops,
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
                        &mut ops,
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
                        constants: vec![ql as u32, lora as u32, this.pos as u32, f(hp.rms_eps)],
                        groups: [1, 1, 1],
                        independent: false,
                    });
                    // q [heads][qk] in a; absorbed [heads][lora] in b; mixed in c; heads' values in out.
                    mv(
                        this,
                        &mut ops,
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
                        &mut ops,
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
                        buffers: vec![&s.b, latents, &s.c],
                        constants: vec![
                            lora as u32,
                            (this.pos + 1) as u32,
                            f(1.0 / (hp.qk_head as f32).sqrt()),
                        ],
                        groups: [hp.heads as u32, 1, 1],
                        independent: false,
                    });
                    mv(
                        this,
                        &mut ops,
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
                        &mut ops,
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
            // The FFN mixer, applying the attention's (output in a).
            let c = mix(this, hpr, &mut ops, next_o3, cur, &blk.ffn_mixer, 1, &s.a);
            cur = Some(c);
            match &blk.ffn {
                Ffn::Dense { gu, down } => {
                    let ff = gu.rows / 2;
                    mv(
                        this,
                        &mut ops,
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
                        &mut ops,
                        down,
                        &s.b,
                        0,
                        &s.a,
                        0,
                        down.constants(0, 0, false),
                        false,
                    );
                    next_mode = 1;
                    next_o = &s.a;
                }
                Ffn::Moe {
                    router,
                    bias,
                    gu,
                    down,
                } => {
                    mv(
                        this,
                        &mut ops,
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
                        buffers: vec![&s.c, bias, &s.normed, this.route.buffer(), &s.routed],
                        constants: vec![
                            hp.experts as u32,
                            hp.experts_used as u32,
                            f(hp.expert_scale),
                            n as u32,
                            seq(this.pos, il),
                        ],
                        groups: [1, 1, 1],
                        independent: false,
                    });
                    let routed = submit(this, &mut ops)?;
                    // The shared expert, queued behind the router while the CPU works.
                    let ff = gu.rows / 2;
                    mv(
                        this,
                        &mut ops,
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
                        &mut ops,
                        down,
                        &s.b,
                        0,
                        &s.a,
                        0,
                        down.constants(0, 0, false),
                        false,
                    );
                    let ex = this.experts[il].as_ref().ok_or("model_state_invalid")?;
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
                            constants: vec![0, gk, (2 * ff) as u32, n as u32, used as u32, 0],
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
                            constants: vec![1, dk, n as u32, ff as u32, used as u32, 0],
                            groups: [(n as u32).div_ceil(rpg), 1, 1],
                            independent: false,
                        });
                    }
                    pending.push(submit(this, &mut ops)?);
                    let t = std::time::Instant::now();
                    this.await_route(seq(this.pos, il), routed, &mut pending)?;
                    timing[0] += t.elapsed().as_secs_f64();
                    let t = std::time::Instant::now();
                    let sum = if std::env::var_os("KUNO_GLM_SKIP_EXPERTS").is_some() {
                        vec![0f32; n]
                    } else {
                        this.cpu_experts(il, ex)?
                    };
                    timing[1] += t.elapsed().as_secs_f64();
                    if let Some(ex) = &this.experts[il] {
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
        let c = mix(this, hpr, &mut ops, next_o3, cur, last, next_mode, next_o);
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
        if let Ok(mut c) = self.counts.lock() {
            for &e in &ids {
                c[il][e] += 1;
            }
        }
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
    hp: &crate::Hparams,
    ops: &mut Vec<Op<'a>>,
    o3: &'a Buffer,
    cur: Option<usize>,
    mixer: &'a Mixer,
    mode: u32,
    o: &'a Buffer,
) -> usize {
    let s = &this.s;
    let n = hp.embd;
    let (src, dst) = match cur {
        None => (this.host_x.buffer(), 0),
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
            this.cpu_out.buffer(),
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
        let names: [(&Kernel, &str); 11] = [
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
    let stage = gpu.readback((4 * n * 4) as u64)?;
    let mut v = vec![0f32; 4 * n];
    gpu.read_floats(x, 0, &stage, &mut v)?;
    Ok((0..n)
        .map(|i| (0..4).map(|h| v[h * n + i]).sum::<f32>() / 4.0)
        .collect())
}
