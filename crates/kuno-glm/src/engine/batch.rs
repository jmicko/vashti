//! A prompt in batches: every block runs a whole batch of tokens at once, entirely on the
//! device, with no host round trips. Products run on cooperative matrices (`mmc.comp`,
//! half-precision inputs, f32 sums; `mm.comp` for the few odd shapes). The routed experts
//! are grouped by expert (`group.comp`) and run from the device's bank, or from the host
//! slots streamed at the bus's full speed through two staging buffers, one piece while the
//! previous one computes (`mec.comp`): every host expert crosses the bus once per batch, so
//! long batches pay for it least. The KDA scan runs the batch's tokens in order inside one
//! dispatch; attention takes each token's own causal span (or the indexer's choice), four
//! heads per workgroup (`mlab.comp`). After the prompt its expert counts (in memory only)
//! warm the banks. Results agree with the CPU reference to about 2e-3 (the half-precision
//! inputs), closer than `step`'s CPU experts.
use super::*;

/// The second dimension of a large one-dimensional grid (as copyoff.comp, glu.comp).
fn grid(groups: usize) -> [u32; 3] {
    let g = groups as u32;
    [g.min(65535), g.div_ceil(65535), 1]
}

/// One source of a block's experts for mec: weights and scales of gate|up and down, words
/// per slot, slot count, the first entry of its order; for host experts, the run of words
/// to copy into its staging buffer first.
struct Source<'a> {
    gu: (&'a Buffer, &'a Buffer, u32, u32, u32, u32),
    down: (&'a Buffer, &'a Buffer, u32, u32, u32, u32),
    slots: u32,
    order_off: u32,
    copy: Option<(&'a Buffer, &'a Buffer, u32, u32)>,
}

/// The shortest prompt worth a batch.
const MIN_BATCH: usize = 32;

impl<'m> Engine<'m> {
    /// Runs `tokens` at the next positions, a batch at a time (or one at a time when the
    /// batch is 1, or a second device holds experts); returns the most likely next token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        let sequential = self.batch < 2
            || self.e.b.mec.is_none()
            // mlab.comp's shape: latents of 512, heads in fours.
            || self.model.hp.kv_lora != 512
            || !self.model.hp.heads.is_multiple_of(4)
            || self.experts.iter().flatten().any(|e| e.device2.is_some())
            || self.trace.is_some() && std::env::var_os("KUNO_GLM_TRACE_BATCH").is_none();
        let mut next = Err("prompt_empty");
        if sequential {
            for &t in tokens {
                next = self.step(t);
                next?;
            }
            return next;
        }
        // Equal batches (no short tail); a prompt under MIN_BATCH tokens runs a token at a
        // time (a batch streams every host expert across the bus, whatever its length).
        let count = tokens.len().div_ceil(self.batch);
        let size = tokens.len().div_ceil(count.max(1)).max(1);
        for chunk in tokens.chunks(size) {
            if chunk.len() < MIN_BATCH {
                for &t in chunk {
                    next = self.step(t);
                    next?;
                }
            } else {
                next = self.run_batch(chunk);
                next?;
            }
        }
        self.warm_banks()?;
        next
    }

    /// After a prompt: up to KUNO_GLM_WARM (default 16) rounds of trades, one per block per
    /// round, bringing the experts the prompt used most onto the device (its counts, in
    /// memory only, are in the banks' scores).
    fn warm_banks(&mut self) -> Result<()> {
        let rounds: usize = std::env::var("KUNO_GLM_WARM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let margin = std::env::var("KUNO_GLM_TRADE_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.0f32);
        let started = std::time::Instant::now();
        let mut total = 0;
        for _ in 0..rounds {
            let this = &*self;
            let mut ops: Vec<Op> = Vec::new();
            let mut planned = Vec::new();
            for (il, ex) in this.experts.iter().enumerate() {
                let Some(ex) = ex else { continue };
                if planned.len() >= this.trade_stage.len() {
                    break;
                }
                if let Some(t) = ex.best_trade(il, margin) {
                    this.trade_ops(ex, &t, &this.trade_stage[planned.len()], &mut ops)?;
                    planned.push(t);
                }
            }
            if planned.is_empty() {
                break;
            }
            submit(this, &mut ops)?.wait()?;
            drop(ops);
            for t in &planned {
                self.experts[t.block]
                    .as_mut()
                    .ok_or("model_state_invalid")?
                    .settle(t);
            }
            total += planned.len();
        }
        if std::env::var_os("KUNO_GLM_TIMING").is_some() {
            eprintln!(
                "warmed the banks: {total} trades in {:.2} s",
                started.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    /// Clears the recurrent state and the position (a new sequence).
    pub fn reset(&mut self) -> Result<()> {
        for blk in &self.blocks {
            if let Attention::Kda { hist, state, .. } = &blk.attention {
                for b in [hist, state] {
                    self.gpu.upload(b, 0, &vec![0u8; b.bytes() as usize])?;
                }
            }
        }
        self.pos = 0;
        Ok(())
    }

    /// Each MoE block's order for mme: its device slots' experts, then its host slots'.
    fn write_order(&self) -> Result<()> {
        let e = self.model.hp.experts;
        let mut order = vec![u32::MAX; self.model.hp.layers * 2 * e];
        for (il, ex) in self.experts.iter().enumerate() {
            let Some(ex) = ex else { continue };
            let o = &mut order[il * 2 * e..][..2 * e];
            for (k, r) in ex.resident.iter().enumerate() {
                o[k] = r.map_or(u32::MAX, |x| x as u32);
            }
            let cap = ex.resident.len();
            for (x, h) in ex.host_slot.iter().enumerate() {
                if let Some(h) = h {
                    o[cap + *h as usize] = x as u32;
                }
            }
        }
        self.gpu.write_mapped(
            &self.order,
            0,
            &kuno_image::engine::gpu::floats(bytes_of(&order)),
        )
    }

    /// Block `il`'s expert sources: the device's bank, then the host slots in pieces that
    /// alternate between the two staging buffers.
    fn sources<'a>(&'a self, il: usize, ex: &'a BlockExperts) -> Vec<Source<'a>> {
        let base = (il * 2 * self.model.hp.experts) as u32;
        let mut out = Vec::new();
        if let Some(pool) = &ex.device {
            let st = pool.stride.map(|v| v as u32);
            out.push(Source {
                gu: (&pool.gu_w, &pool.gu_s, 0, st[0], 0, st[1]),
                down: (&pool.down_w, &pool.down_s, 0, st[2], 0, st[3]),
                slots: pool.capacity as u32,
                order_off: base,
                copy: None,
            });
        }
        let st = ex.stride.map(|v| v as u32);
        let rec: u32 = st.iter().sum();
        let cap = ex.resident.len() as u32;
        let mut k = 0;
        for (c, chunk) in ex.arena.iter().enumerate() {
            let b = chunk.buffer();
            let slots = ((b.bytes() / (u64::from(rec) * 4)) as usize).min(ex.per_chunk);
            for first in (0..slots).step_by(self.piece) {
                let count = self.piece.min(slots - first);
                let to = &self.stage[k % 2];
                k += 1;
                out.push(Source {
                    gu: (to, to, 0, rec, st[0], rec),
                    down: (to, to, st[0] + st[1], rec, st[0] + st[1] + st[2], rec),
                    slots: count as u32,
                    order_off: base + cap + (c * ex.per_chunk + first) as u32,
                    copy: Some((b, to, first as u32 * rec, count as u32 * rec)),
                });
            }
        }
        out
    }

    fn run_batch(&mut self, tokens: &[u32]) -> Result<u32> {
        let hp = self.model.hp.clone();
        let (n, t) = (hp.embd, tokens.len());
        if self.pos + t > self.max_positions {
            return Err("context_full");
        }
        let started = std::time::Instant::now();
        let mut x4 = vec![0f32; t * 4 * n];
        let mut row = vec![0f32; n];
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed.decode_row(tok as usize, &mut row)?;
            for h in 0..4 {
                x4[(i * 4 + h) * n..][..n].copy_from_slice(&row);
            }
        }
        self.gpu.write_mapped(&self.host_x, 0, &x4)?;
        self.write_order()?;
        let this = &*self;
        let s = &this.s;
        let pos = this.pos;
        let tu = t as u32;
        let mut ops: Vec<Op> = Vec::new();
        let mut pending: Vec<Passing> = Vec::new();
        let mut traces = Vec::new();
        let mut cur: Option<usize> = None;
        // The previous sublayer's output for the mixer: mode, o (stride), o2 (stride).
        let mut next: (u32, &Buffer, u32, &Buffer, u32) = (0, &s.out, 0, &s.zeros, 0);
        let (used, e_count, ff) = (hp.experts_used, hp.experts, hp.ff_expert);
        // The lists' capacity (as allocated).
        let lp = this.batch * used + 8 * e_count;
        for (il, blk) in this.blocks.iter().enumerate() {
            let c = bmix(this, &hp, &mut ops, t, cur, &blk.attn, next);
            cur = Some(c);
            if this.trace.is_some() && il > 0 {
                pending.push(submit(this, &mut ops)?);
                for p in pending.drain(..) {
                    p.wait()?;
                }
                traces.push(self_mean_at(&this.gpu, &s.x[c], n, t - 1)?);
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
                    let sr = small.rows;
                    mm(
                        this,
                        &mut ops,
                        qkv,
                        &s.normed,
                        0,
                        n,
                        &s.a,
                        0,
                        3 * d,
                        t,
                        (0, 0),
                    );
                    mm(
                        this,
                        &mut ops,
                        small,
                        &s.normed,
                        0,
                        n,
                        &s.c,
                        0,
                        sr,
                        t,
                        (0, 0),
                    );
                    mm(
                        this,
                        &mut ops,
                        fg_b,
                        &s.c,
                        0,
                        sr,
                        &s.b,
                        0,
                        2 * d,
                        t,
                        (d, 128),
                    );
                    ops.push(Op {
                        kernel: &this.e.b.kda,
                        buffers: vec![&s.a, conv, hist, &s.b, &s.c, params, state, &s.out],
                        constants: vec![
                            d as u32,
                            f(hp.gate_lower_bound),
                            f(hp.rms_eps),
                            256,
                            tu,
                            (3 * d) as u32,
                            (2 * d) as u32,
                            sr as u32,
                            d as u32,
                        ],
                        groups: [hp.heads as u32, 1, 1],
                        independent: false,
                    });
                    mm(this, &mut ops, out, &s.out, 0, d, &s.a, 0, n, t, (0, 0));
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
                    let qa = ql + lora;
                    mm(
                        this,
                        &mut ops,
                        idx_kg,
                        &s.normed,
                        0,
                        n,
                        &s.ia,
                        0,
                        288,
                        t,
                        (0, 0),
                    );
                    mm(
                        this,
                        &mut ops,
                        idx_proj,
                        &s.normed,
                        0,
                        n,
                        &s.ia,
                        256,
                        288,
                        t,
                        (0, 0),
                    );
                    mm(this, &mut ops, q_a, &s.normed, 0, n, &s.c, 0, qa, t, (0, 0));
                    mm(
                        this,
                        &mut ops,
                        kv_a,
                        &s.normed,
                        0,
                        n,
                        &s.c,
                        ql,
                        qa,
                        t,
                        (0, 0),
                    );
                    ops.push(Op {
                        kernel: &this.e.b.dsanorm,
                        buffers: vec![&s.c, norms, &s.out, latents],
                        constants: vec![
                            ql as u32,
                            lora as u32,
                            pos as u32,
                            f(hp.rms_eps),
                            qa as u32,
                            ql as u32,
                        ],
                        groups: [1, 1, tu],
                        independent: false,
                    });
                    for phase in [1, 2] {
                        ops.push(Op {
                            kernel: &this.e.b.idxstore,
                            buffers: vec![&s.ia, idx_params, idx_keys, idx_pooled],
                            constants: vec![pos as u32, f(1e-6), 288, phase],
                            groups: [1, 1, tu],
                            independent: false,
                        });
                    }
                    let kpool = hp.indexer_kpool;
                    let keep = hp.indexer_top_k / kpool;
                    let last_pools = (pos + t) / kpool;
                    let score_st = (this.max_positions / 4).max(4);
                    let idx_st = hp.indexer_top_k + kpool;
                    if last_pools > keep {
                        mm(
                            this,
                            &mut ops,
                            idx_qb,
                            &s.out,
                            0,
                            ql,
                            &s.ib,
                            0,
                            4096,
                            t,
                            (0, 0),
                        );
                        ops.push(Op {
                            kernel: &this.e.b.idxscore,
                            buffers: vec![&s.ib, &s.ia, idx_pooled, &s.iscore],
                            constants: vec![
                                0,
                                f(1.0 / ((128 * 32) as f32).sqrt()),
                                256,
                                pos as u32,
                                keep as u32,
                                kpool as u32,
                                288,
                                score_st as u32,
                            ],
                            groups: [(last_pools as u32).div_ceil(128), 1, tu],
                            independent: false,
                        });
                        ops.push(Op {
                            kernel: &this.e.b.idxselect,
                            buffers: vec![&s.iscore, &s.iidx],
                            constants: vec![
                                pos as u32,
                                keep as u32,
                                kpool as u32,
                                score_st as u32,
                                idx_st as u32,
                            ],
                            groups: [1, 1, tu],
                            independent: false,
                        });
                    }
                    let (hq, hl, hv) =
                        (hp.heads * hp.qk_head, hp.heads * lora, hp.heads * hp.v_head);
                    mm(this, &mut ops, q_b, &s.out, 0, ql, &s.a, 0, hq, t, (0, 0));
                    mm(
                        this,
                        &mut ops,
                        k_b,
                        &s.a,
                        0,
                        hq,
                        &s.b,
                        0,
                        hl,
                        t,
                        (lora, hp.qk_head),
                    );
                    ops.push(Op {
                        kernel: &this.e.b.mla,
                        buffers: vec![&s.b, latents, &s.c, &s.iidx],
                        constants: vec![
                            lora as u32,
                            f(1.0 / (hp.qk_head as f32).sqrt()),
                            pos as u32,
                            hp.indexer_top_k as u32,
                            kpool as u32,
                            hl as u32,
                            hl as u32,
                            idx_st as u32,
                        ],
                        groups: [(hp.heads / 4) as u32, tu, 1],
                        independent: false,
                    });
                    mm(
                        this,
                        &mut ops,
                        v_b,
                        &s.c,
                        0,
                        hl,
                        &s.out,
                        0,
                        hv,
                        t,
                        (hp.v_head, lora),
                    );
                    mm(this, &mut ops, out, &s.out, 0, hv, &s.a, 0, n, t, (0, 0));
                }
            }
            let c = bmix(
                this,
                &hp,
                &mut ops,
                t,
                cur,
                &blk.ffn_mixer,
                (1, &s.a, n as u32, &s.zeros, 0),
            );
            cur = Some(c);
            match &blk.ffn {
                Ffn::Dense { gu, down } => {
                    let ffd = gu.rows / 2;
                    mm(
                        this,
                        &mut ops,
                        gu,
                        &s.normed,
                        0,
                        n,
                        &s.out,
                        0,
                        2 * ffd,
                        t,
                        (0, 0),
                    );
                    ops.push(Op {
                        kernel: &this.e.glu,
                        buffers: vec![&s.out, &s.b],
                        constants: vec![ffd as u32, f(hp.clamp_shared[il]), tu],
                        groups: grid((t * ffd).div_ceil(256)),
                        independent: false,
                    });
                    mm(this, &mut ops, down, &s.b, 0, ffd, &s.a, 0, n, t, (0, 0));
                    next = (1, &s.a, n as u32, &s.zeros, 0);
                }
                Ffn::Moe {
                    router,
                    bias,
                    gu,
                    down,
                } => {
                    mm(
                        this,
                        &mut ops,
                        router,
                        &s.normed,
                        0,
                        n,
                        &s.c,
                        0,
                        e_count,
                        t,
                        (0, 0),
                    );
                    ops.push(Op {
                        kernel: &this.e.b.route,
                        buffers: vec![&s.c, bias, &s.normed, this.route.buffer(), &s.routed],
                        constants: vec![e_count as u32, used as u32, f(hp.expert_scale)],
                        groups: [1, 1, tu],
                        independent: false,
                    });
                    ops.push(Op {
                        kernel: &this.e.b.group,
                        buffers: vec![&s.routed, &s.lists, this.usage.buffer()],
                        constants: vec![
                            tu,
                            used as u32,
                            e_count as u32,
                            (il * e_count) as u32,
                            lp as u32,
                        ],
                        groups: [1, 1, 1],
                        independent: false,
                    });
                    // The shared expert.
                    let ffs = gu.rows / 2;
                    mm(
                        this,
                        &mut ops,
                        gu,
                        &s.normed,
                        0,
                        n,
                        &s.out,
                        0,
                        2 * ffs,
                        t,
                        (0, 0),
                    );
                    ops.push(Op {
                        kernel: &this.e.glu,
                        buffers: vec![&s.out, &s.b],
                        constants: vec![ffs as u32, f(hp.clamp_shared[il]), tu],
                        groups: grid((t * ffs).div_ceil(256)),
                        independent: false,
                    });
                    mm(this, &mut ops, down, &s.b, 0, ffs, &s.a, 0, n, t, (0, 0));
                    // The routed experts, from each source (disjoint outputs: no barriers
                    // between sources).
                    let ex = this.experts[il].as_ref().ok_or("model_state_invalid")?;
                    let (gk, dk) = (experts::kind(ex.gate_kind)?, experts::kind(ex.down_kind)?);
                    let sources = this.sources(il, ex);
                    let k = (gk, dk, used, e_count, ff, t);
                    let limit = hp.clamp_expert[il];
                    // The bank's experts beside the first piece's copy; then per piece its
                    // gate|up beside the next piece's copy, and its down.
                    let staged: Vec<&Source> =
                        sources.iter().filter(|s| s.copy.is_some()).collect();
                    let bank = sources.iter().find(|s| s.copy.is_none());
                    if let Some(c) = staged.first().and_then(|s| s.copy) {
                        let mut op = copy_op(this, c);
                        op.independent = bank.is_some();
                        ops.push(op);
                    }
                    if let Some(b) = bank {
                        ops.push(mec_op(this, b, 0, k, limit, false)?);
                        ops.push(mec_op(this, b, 1, k, limit, false)?);
                    }
                    for (i, src) in staged.iter().enumerate() {
                        let copy = staged.get(i + 1).and_then(|s| s.copy);
                        ops.push(mec_op(this, src, 0, k, limit, copy.is_some())?);
                        if let Some(c) = copy {
                            ops.push(copy_op(this, c));
                        }
                        ops.push(mec_op(this, src, 1, k, limit, false)?);
                    }
                    ops.push(Op {
                        kernel: &this.e.b.combine,
                        buffers: vec![&s.gd, &s.c, &s.routed, &s.lists],
                        constants: vec![
                            n as u32,
                            used as u32,
                            (t * n) as u32,
                            e_count as u32,
                            lp as u32,
                        ],
                        groups: grid((t * n).div_ceil(256)),
                        independent: false,
                    });
                    next = (2, &s.a, n as u32, &s.c, n as u32);
                }
            }
            pending.push(submit(this, &mut ops)?);
        }
        // The last block's FFN mixer, then the last token's norm, head and choice.
        let c = bmix(this, &hp, &mut ops, t, cur, &this.blocks[0].attn, next);
        if this.trace.is_some() {
            pending.push(submit(this, &mut ops)?);
            for p in pending.drain(..) {
                p.wait()?;
            }
            traces.push(self_mean_at(&this.gpu, &s.x[c], n, t - 1)?);
        }
        ops.push(Op {
            kernel: &this.e.copyoff,
            buffers: vec![&s.x[c], &s.out],
            constants: vec![(4 * n) as u32, ((t - 1) * 4 * n) as u32, 0],
            groups: grid((4 * n).div_ceil(256)),
            independent: false,
        });
        ops.push(Op {
            kernel: &this.e.fin,
            buffers: vec![&s.out, &this.out_norm, &s.normed],
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
        let mut tok = [0f32; 1];
        this.gpu.read_mapped(&this.token, 0, &mut tok)?;
        drop(ops);
        // The batch's choices into the banks' scores.
        let e = hp.experts;
        let mut counts = vec![0f32; hp.layers * e];
        self.gpu.read_mapped(&self.usage, 0, &mut counts)?;
        let last = (self.pos + t - 1) as u64;
        for (il, ex) in self.experts.iter().enumerate() {
            if let Some(ex) = ex {
                let c: Vec<u32> = counts[il * e..][..e].iter().map(|v| v.to_bits()).collect();
                ex.add_batch(&c, last, t);
            }
        }
        if let Some(tr) = &mut self.trace {
            *tr = traces;
        }
        if std::env::var_os("KUNO_GLM_TIMING").is_some() {
            let s = started.elapsed().as_secs_f64();
            eprintln!(
                "batch of {t} at {}: {:.2} s ({:.1} tokens/s)",
                self.pos,
                s,
                t as f64 / s
            );
        }
        self.pos += t;
        Ok(tok[0].to_bits())
    }
}

/// The routed experts of `src` (mode 0 gate|up, 1 down with the SwiGLU of the gate|up
/// results, clamped at `limit`); `k`: the gate and down kinds, experts per token, experts,
/// ff, tokens.
fn mec_op<'a>(
    this: &'a Engine,
    src: &Source<'a>,
    mode: u32,
    k: (u32, u32, usize, usize, usize, usize),
    limit: f32,
    independent: bool,
) -> Result<Op<'a>> {
    let (gk, dk, used, e_count, ff, tokens) = k;
    let s = &this.s;
    let n = this.model.hp.embd;
    let (w, sc, wb, wst, sb, sst) = if mode == 0 { src.gu } else { src.down };
    let (r, cols, kind, x, y) = if mode == 0 {
        (2 * ff, n, gk, &s.normed, &s.ge)
    } else {
        (n, ff, dk, &s.ge, &s.gd)
    };
    Ok(Op {
        kernel: this.e.b.mec.as_ref().ok_or("gpu_unsupported")?,
        buffers: vec![w, sc, x, y, this.order.buffer(), &s.lists, &this.e.tables],
        constants: vec![
            mode,
            kind,
            r as u32,
            cols as u32,
            used as u32,
            e_count as u32,
            src.order_off,
            wb,
            wst,
            sb,
            sst,
            f(limit),
        ],
        groups: [
            (r as u32).div_ceil(128),
            src.slots,
            (tokens as u32).div_ceil(64),
        ],
        independent,
    })
}

/// A run of `words` words from `from` (at word `at`) to the start of `to`, 16 bytes at a
/// time when both are multiples of 4.
fn copy_op<'a>(this: &'a Engine, c: (&'a Buffer, &'a Buffer, u32, u32)) -> Op<'a> {
    if !c.2.is_multiple_of(4) || !c.3.is_multiple_of(4) {
        return Op {
            kernel: &this.e.copyoff,
            buffers: vec![c.0, c.1],
            constants: vec![c.3, c.2, 0],
            groups: grid((c.3 as usize).div_ceil(256)),
            independent: false,
        };
    }
    Op {
        kernel: &this.e.b.copy4,
        buffers: vec![c.0, c.1],
        constants: vec![c.3 / 4, c.2 / 4, 0],
        groups: grid((c.3 as usize / 4).div_ceil(256)),
        independent: false,
    }
}

/// Records the batched `m x` for `tokens` tokens: token i's input at i*x_st + x_off, its
/// output at i*y_st + y_off; `group` (rows per input group, its stride) as `Mat::grouped`.
#[allow(clippy::too_many_arguments)]
fn mm<'a>(
    this: &'a Engine,
    ops: &mut Vec<Op<'a>>,
    m: &'a Mat,
    x: &'a Buffer,
    x_off: usize,
    x_st: usize,
    y: &'a Buffer,
    y_off: usize,
    y_st: usize,
    tokens: usize,
    group: (usize, usize),
) {
    debug_assert!(x_off.is_multiple_of(4) && x_st.is_multiple_of(4) && group.1.is_multiple_of(4));
    if let Some(mmc) = &this.e.b.mmc
        && m.rows.is_multiple_of(128)
        && m.cols.is_multiple_of(32)
        && (m.kind == 0 || m.kind == 3)
        && group.0.is_multiple_of(128)
    {
        ops.push(Op {
            kernel: mmc,
            buffers: vec![&m.w, &m.s, x, y],
            constants: vec![
                m.rows as u32,
                m.cols as u32,
                m.kind,
                x_off as u32,
                y_off as u32,
                group.0 as u32,
                group.1 as u32,
                x_st as u32,
                y_st as u32,
                tokens as u32,
            ],
            groups: [(m.rows / 128) as u32, (tokens as u32).div_ceil(64), 1],
            independent: false,
        });
        return;
    }
    let rows = 128 / this.k.lanes;
    ops.push(Op {
        kernel: &this.e.b.mm,
        buffers: vec![&m.w, &m.s, x, y],
        constants: vec![
            m.rows as u32,
            m.cols as u32,
            m.kind,
            x_off as u32,
            y_off as u32,
            0,
            0,
            group.0 as u32,
            group.1 as u32,
            x_st as u32,
            y_st as u32,
            tokens as u32,
        ],
        groups: [
            (m.rows as u32).div_ceil(rows),
            (tokens as u32).div_ceil(128),
            1,
        ],
        independent: false,
    });
}

/// The batched mixer (as `mix`): `prev` is the previous sublayer's mode, output o and its
/// stride, and the routed experts' sum o2 and its stride (mode 2).
fn bmix<'a>(
    this: &'a Engine,
    hp: &crate::Hparams,
    ops: &mut Vec<Op<'a>>,
    tokens: usize,
    cur: Option<usize>,
    mixer: &'a Mixer,
    prev: (u32, &'a Buffer, u32, &'a Buffer, u32),
) -> usize {
    let s = &this.s;
    let n = hp.embd;
    let (src, dst) = match cur {
        None => (this.host_x.buffer(), 0),
        Some(i) => (&s.x[i], 1 - i),
    };
    let mode = if cur.is_none() { 3 } else { prev.0 };
    let xn = &s.x[dst];
    ops.push(Op {
        kernel: &this.e.b.hcdot,
        buffers: vec![
            &mixer.function.w,
            &mixer.function.s,
            src,
            &s.part,
            &s.coef,
            prev.1,
            xn,
            prev.3,
            &s.zeros,
        ],
        constants: vec![n as u32, mode, 0, prev.2, prev.4, 0],
        groups: [32, 1, tokens as u32],
        independent: false,
    });
    ops.push(Op {
        kernel: &this.e.b.hcmix,
        buffers: vec![&s.part, xn, &mixer.params, &mixer.norm, &s.coef, &s.normed],
        constants: vec![
            n as u32,
            hp.sinkhorn as u32,
            f(hp.rms_eps),
            f(hp.hc_eps),
            32,
        ],
        groups: [1, 1, tokens as u32],
        independent: false,
    });
    dst
}

#[cfg(test)]
mod tests {
    use kuno_image::engine::gpu::{Gpu, Select, Step, bytes_of};

    use crate::Model;
    use crate::gpu::{Mat, filled};

    /// mmc (half-precision inputs) against mm (f32) on real weights, every row of a token
    /// tile in use: their difference relative to the largest output.
    #[test]
    #[ignore = "needs an Intel Xe2 GPU and the model"]
    fn cooperative_products_match_f32() {
        let Some(dir) = std::env::var_os("KUNO_GLM_MODEL") else {
            return;
        };
        let mut shards: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
            .collect();
        shards.sort();
        let model = Model::open(&shards).unwrap();
        let pci = std::env::var("KUNO_GLM_GPU").unwrap_or_else(|_| "0000:2d:00.0".into());
        let gpu = Gpu::open(Select::Pci(pci)).unwrap();
        let lanes = Some(16);
        let mm = gpu
            .kernel(include_bytes!("../../shaders/mm.spv"), 4, 12, lanes)
            .unwrap();
        let mmc = gpu
            .kernel(include_bytes!("../../shaders/mmc.spv"), 4, 10, lanes)
            .unwrap();
        for (name, tokens) in [
            ("blk.5.attn_q.weight", 64usize),
            ("blk.5.attn_q.weight", 100),
            ("blk.3.attn_q_b.weight", 77),
            ("blk.5.attn_output.weight", 64),
        ] {
            let m = model.matrix(name).unwrap();
            let g = Mat::upload(&gpu, &m).unwrap();
            let x: Vec<f32> = (0..tokens * m.cols)
                .map(|i| ((i * 7919 + i / m.cols * 13) % 251) as f32 / 61.0 - 2.0)
                .collect();
            let xb = filled(&gpu, bytes_of(&x)).unwrap();
            let out = |k: &kuno_image::engine::gpu::Kernel, c: Vec<u32>, groups: [u32; 3]| {
                let y = gpu.buffer((128 * m.rows * 4) as u64).unwrap();
                let b = [&g.w, &g.s, &xb, &y];
                gpu.pass(&[Step {
                    kernel: k,
                    buffers: &b,
                    constants: &c,
                    groups,
                }])
                .unwrap();
                let stage = gpu.readback((tokens * m.rows * 4) as u64).unwrap();
                let mut v = vec![0f32; tokens * m.rows];
                gpu.read_floats(&y, 0, &stage, &mut v).unwrap();
                v
            };
            let (r, c, t) = (m.rows as u32, m.cols as u32, tokens as u32);
            let want = out(
                &mm,
                vec![r, c, g.kind, 0, 0, 0, 0, 0, 0, c, r, t],
                [r.div_ceil(8), t.div_ceil(128), 1],
            );
            let got = out(
                &mmc,
                vec![r, c, g.kind, 0, 0, 0, 0, c, r, t],
                [r / 128, t.div_ceil(64), 1],
            );
            let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let mut worst = (0f32, 0usize);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                let d = (a - b).abs();
                if d > worst.0 {
                    worst = (d, i);
                }
            }
            eprintln!(
                "{name} x{tokens}: worst {:.2e} of {scale:.2e} at token {} row {}",
                worst.0,
                worst.1 / m.rows,
                worst.1 % m.rows
            );
            assert!(worst.0 <= 1e-2 * scale, "{name}");
        }
    }
}
