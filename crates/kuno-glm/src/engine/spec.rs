//! Speculative decoding with the model's NextN block (`blk.45`, multi-token prediction): a
//! plain-residual DSA block that reads a token's embedding and the main model's final
//! hidden state at the position before (`h`, the output norm of the streams' mean) and
//! predicts the token after. Drafting runs it a token at a time (its own output state
//! feeding the next draft); the main model then checks the current token and the drafts in
//! one small batch (`run_small`), and the longest prefix whose drafts match its own greedy
//! choices is kept: the output is the plain greedy decode's. The KDA blocks' states are
//! snapshotted after each of the batch's tokens, so a rejected draft rolls back; the
//! position-indexed caches are simply overwritten later. The NextN block's own caches
//! follow every position the main model processes (`mtp_follow`: only its cache writes,
//! no attention or experts), with the true `h`.
//!
//! The NextN block's routed experts (Q2_K/Q3_K, kept as stored) run on the CPU, which is
//! idle while drafting.
use super::batch::{MIN_BATCH, attention_ops, bmix, bmix3, grid, mm};
use super::*;
use kuno_text::TensorType;
use rayon::prelude::*;

/// Tokens one small batch (a verification, or a short prompt's chunk) runs at most.
pub(super) const SMALL: usize = MIN_BATCH;

pub(super) struct Mtp {
    pub(super) eh: Mat,
    /// enorm, hnorm, attn_norm, ffn_norm, shared head norm: [n] each.
    pub(super) norms: Buffer,
    pub(super) attention: Attention,
    pub(super) router: Mat,
    pub(super) bias: Buffer,
    pub(super) gu: Mat,
    pub(super) down: Mat,
    /// Gate|up and down types and bytes per row (raw GGUF rows).
    pub(super) kinds: (TensorType, TensorType),
    pub(super) row_bytes: (usize, usize),
    /// Every routed expert's raw rows (gate, up, down), once loaded.
    pub(super) experts: std::sync::OnceLock<Vec<Vec<u8>>>,
    /// Device scratch per token: [e | h] normed inputs [2n], the residual, the FFN's input.
    pub(super) cat: Buffer,
    pub(super) res: Buffer,
    pub(super) inp: Buffer,
    pub(super) post: Buffer,
    /// The h inputs [tokens][n], and the main model's h at the last position processed.
    pub(super) h: Buffer,
    pub(super) last: Buffer,
    /// Host-visible: token embeddings [tokens][n]; a draft's routing and input; its CPU sum.
    pub(super) emb: Mapped,
    pub(super) vin: Mapped,
    pub(super) cpu: Mapped,
}

/// A speculative decode's counts.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecStats {
    pub steps: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub draft_s: f64,
    pub trades: usize,
    pub verify_s: f64,
}

impl<'m> Engine<'m> {
    /// Whether the engine drafts with the NextN block (KUNO_GLM_SPEC > 0 and the block in
    /// the file).
    pub fn speculates(&self) -> bool {
        self.mtp.is_some()
    }

    /// Records the NextN block catching up on `t` positions from `pos` (tokens' embeddings
    /// already in its `emb`): its h inputs are its `last` then the main model's `hall`
    /// rows before the last; only its caches are written. `last_row`: the `hall` row that
    /// becomes `last` (the last position kept).
    pub(super) fn mtp_follow<'a>(
        &'a self,
        ops: &mut Vec<Op<'a>>,
        t: usize,
        pos: usize,
        last_row: Option<usize>,
    ) {
        let Some(mtp) = &self.mtp else { return };
        let n = self.model.hp.embd;
        let copy = |ops: &mut Vec<Op<'a>>,
                    src: &'a Buffer,
                    dst: &'a Buffer,
                    words: usize,
                    so: usize,
                    d_o: usize| {
            ops.push(Op {
                kernel: &self.e.copyoff,
                buffers: vec![src, dst],
                constants: vec![words as u32, so as u32, d_o as u32],
                groups: grid(words.div_ceil(256)),
                independent: true,
            });
        };
        copy(ops, &mtp.last, &mtp.h, n, 0, 0);
        if t > 1 {
            copy(ops, &self.hall, &mtp.h, (t - 1) * n, 0, n);
        }
        if let Some(op) = ops.last_mut() {
            op.independent = false;
        }
        self.mtp_front(ops, t);
        attention_ops(
            self,
            ops,
            self.model.hp.layers,
            &mtp.attention,
            t,
            pos,
            0,
            false,
        );
        if let Some(r) = last_row {
            copy(ops, &self.hall, &mtp.last, n, r * n, 0);
            if let Some(op) = ops.last_mut() {
                op.independent = false;
            }
        }
    }

    /// The NextN block up to its attention's input (in normed): the normed embedding and h
    /// side by side, eh_proj into `res`, attn_norm.
    fn mtp_front<'a>(&'a self, ops: &mut Vec<Op<'a>>, t: usize) {
        let Some(mtp) = &self.mtp else { return };
        let hp = &self.model.hp;
        let n = hp.embd;
        let eps = f(hp.rms_eps);
        let tu = t as u32;
        let rms = |ops: &mut Vec<Op<'a>>,
                   x: &'a Buffer,
                   x_st: usize,
                   o: &'a Buffer,
                   o_st: usize,
                   o_off: usize,
                   w: usize,
                   indep: bool| {
            ops.push(Op {
                kernel: &self.e.b.rmsb,
                buffers: vec![x, &mtp.norms, o],
                constants: vec![
                    n as u32,
                    eps,
                    x_st as u32,
                    o_st as u32,
                    o_off as u32,
                    (w * n) as u32,
                ],
                groups: [1, 1, tu],
                independent: indep,
            });
        };
        rms(ops, mtp.emb.buffer(), n, &mtp.cat, 2 * n, 0, 0, true);
        rms(ops, &mtp.h, n, &mtp.cat, 2 * n, n, 1, false);
        mm(
            self,
            ops,
            &mtp.eh,
            &mtp.cat,
            0,
            2 * n,
            &mtp.res,
            0,
            n,
            t,
            (0, 0),
        );
        rms(ops, &mtp.res, n, &self.s.normed, n, 0, 2, false);
    }

    /// One draft: the NextN block on `token` at `pos` with h from its `h` row 0; its output
    /// state goes back to `h` row 0 (the next draft's input). Returns its greedy choice.
    fn mtp_draft(&self, token: u32, pos: usize) -> Result<u32> {
        let mtp = self.mtp.as_ref().ok_or("model_state_invalid")?;
        let hp = &self.model.hp;
        let (n, used, e) = (hp.embd, hp.experts_used, hp.experts);
        let s = &self.s;
        let mut row = vec![0f32; n];
        self.embed.decode_row(token as usize, &mut row)?;
        self.gpu.write_mapped(&mtp.emb, 0, &row)?;
        let mut ops: Vec<Op> = Vec::new();
        self.mtp_front(&mut ops, 1);
        attention_ops(self, &mut ops, hp.layers, &mtp.attention, 1, pos, 0, true);
        ops.push(sum3_op(self, &s.a, &s.zeros, 0, &mtp.res, &mtp.inp));
        ops.push(Op {
            kernel: &self.e.b.rmsb,
            buffers: vec![&mtp.inp, &mtp.norms, &s.normed],
            constants: vec![
                n as u32,
                f(hp.rms_eps),
                n as u32,
                n as u32,
                0,
                (3 * n) as u32,
            ],
            groups: [1, 1, 1],
            independent: false,
        });
        mm(
            self,
            &mut ops,
            &mtp.router,
            &s.normed,
            0,
            n,
            &s.c,
            0,
            e,
            1,
            (0, 0),
        );
        ops.push(Op {
            kernel: &self.e.b.route,
            buffers: vec![&s.c, &mtp.bias, &s.normed, self.route.buffer(), &s.routed],
            constants: vec![e as u32, used as u32, f(hp.expert_scale)],
            groups: [1, 1, 1],
            independent: false,
        });
        for (src, words, at) in [(&s.routed, 2 * used, 0), (&s.normed, n, 2 * used)] {
            ops.push(Op {
                kernel: &self.e.copyoff,
                buffers: vec![src, mtp.vin.buffer()],
                constants: vec![words as u32, 0, at as u32],
                groups: grid(words.div_ceil(256)),
                independent: true,
            });
        }
        let ffs = mtp.gu.rows / 2;
        mm(
            self,
            &mut ops,
            &mtp.gu,
            &s.normed,
            0,
            n,
            &s.out,
            0,
            2 * ffs,
            1,
            (0, 0),
        );
        ops.push(Op {
            kernel: &self.e.glu,
            buffers: vec![&s.out, &s.b],
            constants: vec![
                ffs as u32,
                f(*hp.clamp_shared.get(hp.layers).unwrap_or(&0.0)),
                1,
            ],
            groups: grid(ffs.div_ceil(256)),
            independent: false,
        });
        mm(
            self,
            &mut ops,
            &mtp.down,
            &s.b,
            0,
            ffs,
            &s.a,
            0,
            n,
            1,
            (0, 0),
        );
        submit(self, &mut ops)?.wait()?;
        let mut r = vec![0f32; 2 * used + n];
        self.gpu.read_mapped(&mtp.vin, 0, &mut r)?;
        let ids: Vec<usize> = r[..used].iter().map(|v| v.to_bits() as usize).collect();
        let sum = self.mtp_experts(&ids, &r[used..2 * used], &r[2 * used..])?;
        self.gpu.write_mapped(&mtp.cpu, 0, &sum)?;
        ops.push(sum3_op(
            self,
            &s.a,
            mtp.cpu.buffer(),
            n as u32,
            &mtp.inp,
            &mtp.post,
        ));
        ops.push(Op {
            kernel: &self.e.b.rmsb,
            buffers: vec![&mtp.post, &mtp.norms, &mtp.h],
            constants: vec![
                n as u32,
                f(hp.rms_eps),
                n as u32,
                n as u32,
                0,
                (4 * n) as u32,
            ],
            groups: [1, 1, 1],
            independent: false,
        });
        mv(
            self,
            &mut ops,
            &self.output,
            &mtp.h,
            0,
            &s.logits,
            0,
            self.output.constants(0, 0, false),
            false,
        );
        ops.push(Op {
            kernel: &self.e.argmax,
            buffers: vec![&s.logits, self.token.buffer()],
            constants: vec![hp.vocab as u32, 0],
            groups: [1, 1, 1],
            independent: false,
        });
        submit(self, &mut ops)?.wait()?;
        let mut tok = [0f32; 1];
        self.gpu.read_mapped(&self.token, 0, &mut tok)?;
        Ok(tok[0].to_bits())
    }

    /// The NextN block's routed experts on the CPU (its raw Q2_K/Q3_K rows), weighted.
    fn mtp_experts(&self, ids: &[usize], w: &[f32], x: &[f32]) -> Result<Vec<f32>> {
        let mtp = self.mtp.as_ref().ok_or("model_state_invalid")?;
        let hp = &self.model.hp;
        let (n, ff) = (hp.embd, hp.ff_expert);
        let raws = mtp.experts.get().ok_or("model_state_invalid")?;
        let (gk, dk) = mtp.kinds;
        let (gr, dr) = mtp.row_bytes;
        let limit = *hp.clamp_expert.get(hp.layers).unwrap_or(&0.0);
        let q = crate::cpu::Q8K::new(x);
        let mut gu = vec![0f32; ids.len() * 2 * ff];
        gu.par_chunks_mut(64).enumerate().for_each(|(task, out)| {
            let row0 = task * 64;
            let (k, r) = (row0 / (2 * ff), row0 % (2 * ff));
            let raw = &raws[ids[k]];
            crate::cpu::matvec_k(gk, &raw[r * gr..][..out.len() * gr], gr, &q, out);
        });
        let hidden: Vec<crate::cpu::Q8K> = gu
            .chunks_exact(2 * ff)
            .map(|g| {
                let h: Vec<f32> = (0..ff)
                    .map(|i| {
                        let (a, b) = if limit > 1e-6 {
                            (g[i].min(limit), g[ff + i].clamp(-limit, limit))
                        } else {
                            (g[i], g[ff + i])
                        };
                        a / (1.0 + (-a).exp()) * b
                    })
                    .collect();
                crate::cpu::Q8K::new(&h)
            })
            .collect();
        let mut d = vec![0f32; ids.len() * n];
        d.par_chunks_mut(64).enumerate().for_each(|(task, out)| {
            let row0 = task * 64;
            let (k, r) = (row0 / n, row0 % n);
            let raw = &raws[ids[k]][2 * ff * gr..];
            crate::cpu::matvec_k(dk, &raw[r * dr..][..out.len() * dr], dr, &hidden[k], out);
        });
        let mut sum = vec![0f32; n];
        for (k, wk) in w.iter().enumerate() {
            for (s, v) in sum.iter_mut().zip(&d[k * n..][..n]) {
                *s += wk * v;
            }
        }
        Ok(sum)
    }

    /// Loads the NextN block's experts (raw rows in memory) if not yet.
    fn load_mtp(&self) -> Result<()> {
        let Some(mtp) = &self.mtp else { return Ok(()) };
        if mtp.experts.get().is_some() {
            return Ok(());
        }
        let raws = experts::read_block_threads(self.model, self.model.hp.layers, 8)?;
        let _ = mtp.experts.set(raws);
        Ok(())
    }

    /// The main model on `tokens` (at most SMALL) at the next positions, with the routed
    /// experts the device does not hold on the CPU (as `step`) and f32 products: each
    /// position's greedy choice. KDA states after the first `snaps` tokens are kept for
    /// `restore`; the NextN block follows (its `last` from the last token).
    pub(super) fn run_small(&mut self, tokens: &[u32], snaps: u32) -> Result<Vec<u32>> {
        let hp = self.model.hp.clone();
        let (n, t) = (hp.embd, tokens.len());
        if t == 0 || t > SMALL {
            return Err("model_state_invalid");
        }
        if self.pos + t > self.max_positions {
            return Err("context_full");
        }
        self.load_all()?;
        let mut x4 = vec![0f32; t * 4 * n];
        let mut emb = vec![0f32; t * n];
        for (i, &tok) in tokens.iter().enumerate() {
            let row = &mut emb[i * n..][..n];
            self.embed.decode_row(tok as usize, row)?;
            for h in 0..4 {
                x4[(i * 4 + h) * n..][..n].copy_from_slice(row);
            }
        }
        self.gpu.write_mapped(&self.host_x, 0, &x4)?;
        if let Some(mtp) = &self.mtp {
            self.gpu.write_mapped(&mtp.emb, 0, &emb)?;
        }
        self.exact.store(true, std::sync::atomic::Ordering::Relaxed);
        let result = self.small_pass(t, snaps);
        self.exact
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let out = result?;
        // The batch's choices into the banks' scores.
        let e = hp.experts;
        let mut counts = vec![0f32; hp.layers * e];
        self.gpu.read_mapped(&self.usage, 0, &mut counts)?;
        let last = (self.pos + t - 1) as u64;
        for (il, ex) in self.experts.iter().enumerate() {
            if let Some(ex) = ex.get() {
                let c: Vec<u32> = counts[il * e..][..e].iter().map(|v| v.to_bits()).collect();
                ex.add_batch(&c, last, t);
                note_uses(il, e, &c, ex);
            }
        }
        self.pos += t;
        Ok(out)
    }

    fn small_pass(&self, t: usize, snaps: u32) -> Result<Vec<u32>> {
        let this = self;
        let hp = &this.model.hp;
        let s = &this.s;
        let n = hp.embd;
        let pos = this.pos;
        let tu = t as u32;
        let (used, e_count, ff) = (hp.experts_used, hp.experts, hp.ff_expert);
        let lp = this.batch * used + 8 * e_count;
        let mut ops: Vec<Op> = Vec::new();
        let mut pending: Vec<Passing> = Vec::new();
        let mut cur: Option<usize> = None;
        let mut next: (u32, &Buffer, u32, &Buffer, u32) = (0, &s.out, 0, &s.zeros, 0);
        let mut next3: (&Buffer, u32) = (&s.zeros, 0);
        let started = std::time::Instant::now();
        let (mut wait_s, mut cpu_s) = (0f64, 0f64);
        let (mut n_pairs, mut n_unique, mut n_gpu) = (0usize, 0usize, 0usize);
        for (il, blk) in this.blocks.iter().enumerate() {
            let c = bmix3(this, hp, &mut ops, t, cur, &blk.attn, next, next3);
            cur = Some(c);
            attention_ops(this, &mut ops, il, &blk.attention, t, pos, snaps, true);
            let c = bmix(
                this,
                hp,
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
                    next3 = (&s.zeros, 0);
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
                    // The choice and inputs to the host for the CPU's experts.
                    for (src, words, at) in [
                        (&s.routed, t * 2 * used, 0),
                        (&s.normed, t * n, t * 2 * used),
                    ] {
                        ops.push(Op {
                            kernel: &this.e.copyoff,
                            buffers: vec![src, this.vin.buffer()],
                            constants: vec![words as u32, 0, at as u32],
                            groups: grid(words.div_ceil(256)),
                            independent: true,
                        });
                    }
                    let routed = submit(this, &mut ops)?;
                    // The shared expert and the device's experts meanwhile.
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
                    let ex = this.experts[il].get().ok_or("model_state_invalid")?;
                    let rpg = 128 / this.k.lanes;
                    let (gk, dk) = (experts::kind(ex.gate_kind)?, experts::kind(ex.down_kind)?);
                    let limit = hp.clamp_expert[il];
                    // The device's bank as `step` runs it (mve.comp, f32), every token at once.
                    let g0 = [((2 * ff) as u32).div_ceil(rpg), used as u32, tu];
                    let g1 = [(n as u32).div_ceil(rpg), 1, tu];
                    let c0 = |st: [u32; 4]| {
                        vec![
                            0,
                            gk,
                            (2 * ff) as u32,
                            n as u32,
                            used as u32,
                            0,
                            st[0],
                            st[1],
                            st[2],
                            st[3],
                        ]
                    };
                    let c1 = |mode: u32, st: [u32; 4]| {
                        vec![
                            mode,
                            dk,
                            n as u32,
                            ff as u32,
                            used as u32,
                            0,
                            st[0],
                            st[1],
                            st[2],
                            st[3],
                        ]
                    };
                    if let Some(pool) = &ex.device {
                        ops.push(mve_op(
                            this,
                            &pool.gu_w,
                            &pool.gu_s,
                            &s.normed,
                            &s.ge,
                            &pool.slots,
                            c0([0; 4]),
                            g0,
                        ));
                    }
                    pending.push(submit(this, &mut ops)?);
                    let tw = std::time::Instant::now();
                    routed.wait()?;
                    wait_s += tw.elapsed().as_secs_f64();
                    let mut r = vec![0f32; t * (2 * used + n)];
                    this.gpu.read_mapped(&this.vin, 0, &mut r)?;
                    let (choice, xs) = r.split_at(t * 2 * used);
                    let ids: Vec<Vec<usize>> = (0..t)
                        .map(|i| {
                            choice[i * 2 * used..][..used]
                                .iter()
                                .map(|v| v.to_bits() as usize)
                                .collect()
                        })
                        .collect();
                    // KUNO_GLM_GPU_SHARE of the host experts (those most tokens chose) streamed
                    // to the device and run there while the CPU runs the rest.
                    let mut users = std::collections::BTreeMap::<usize, usize>::new();
                    for e in ids.iter().flatten() {
                        if ex.host_slot[*e].is_some() {
                            *users.entry(*e).or_default() += 1;
                        }
                    }
                    let share: f32 = std::env::var("KUNO_GLM_GPU_SHARE")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.25);
                    let stage = this.stage.get(il % this.stage.len().max(1));
                    let mut ranked: Vec<(usize, usize)> =
                        users.iter().map(|(e, u)| (*e, *u)).collect();
                    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    let take = match (stage, this.gslots.get(il)) {
                        (Some(_), Some(_)) => {
                            ((ranked.len() as f32 * share).round() as usize).min(this.piece)
                        }
                        _ => 0,
                    };
                    let gpu_set: Vec<usize> = ranked[..take].iter().map(|x| x.0).collect();
                    if let (Some(stage), Some(map), true) = (stage, this.gslots.get(il), take > 0) {
                        let mut slots = vec![u32::MAX; e_count];
                        let st = ex.stride.map(|v| v as u32);
                        let rec: u32 = st.iter().sum();
                        for (j, &e) in gpu_set.iter().enumerate() {
                            slots[e] = j as u32;
                            let (chunk, at) =
                                ex.host_at(ex.host_slot[e].ok_or("model_state_invalid")?);
                            // 16 bytes at a time where aligned (the bus's full speed).
                            let wide = rec.is_multiple_of(4) && at.is_multiple_of(4);
                            let (kernel, q) = if wide {
                                (&this.e.b.copy4, 4)
                            } else {
                                (&this.e.copyoff, 1)
                            };
                            ops.push(Op {
                                kernel,
                                buffers: vec![chunk.buffer(), stage],
                                constants: vec![rec / q, at as u32 / q, j as u32 * rec / q],
                                groups: grid((rec as usize / q as usize).div_ceil(256)),
                                independent: true,
                            });
                        }
                        if let Some(op) = ops.last_mut() {
                            op.independent = false;
                        }
                        this.gpu.write_mapped(
                            map,
                            0,
                            &kuno_image::engine::gpu::floats(bytes_of(&slots)),
                        )?;
                        ops.push(mve_op(
                            this,
                            stage,
                            stage,
                            &s.normed,
                            &s.ge,
                            map.buffer(),
                            c0([rec, 0, rec, st[0]]),
                            g0,
                        ));
                    }
                    ops.push(Op {
                        kernel: &this.e.glu,
                        buffers: vec![&s.ge, &s.gh],
                        constants: vec![ff as u32, f(limit), (t * used) as u32],
                        groups: grid((t * used * ff).div_ceil(256)),
                        independent: false,
                    });
                    next3 = (&s.zeros, 0);
                    if let Some(pool) = &ex.device {
                        ops.push(mve_op(
                            this,
                            &pool.down_w,
                            &pool.down_s,
                            &s.gh,
                            &s.gd,
                            &pool.slots,
                            c1(1, [0; 4]),
                            g1,
                        ));
                        next3 = (&s.gd, n as u32);
                    }
                    if let (Some(stage), Some(map), true) = (stage, this.gslots.get(il), take > 0) {
                        let st = ex.stride.map(|v| v as u32);
                        let rec: u32 = st.iter().sum();
                        let mode = if ex.device.is_some() { 2 } else { 1 };
                        ops.push(mve_op(
                            this,
                            stage,
                            stage,
                            &s.gh,
                            &s.gd,
                            map.buffer(),
                            c1(mode, [rec, st[0] + st[1], rec, st[0] + st[1] + st[2]]),
                            g1,
                        ));
                        next3 = (&s.gd, n as u32);
                    }
                    pending.push(submit(this, &mut ops)?);
                    // The CPU's share meanwhile.
                    let tc = std::time::Instant::now();
                    let kept: Vec<(Vec<usize>, Vec<f32>)> = (0..t)
                        .map(|i| {
                            let w = &choice[i * 2 * used + used..][..used];
                            ids[i]
                                .iter()
                                .zip(w)
                                .filter(|(e, _)| !gpu_set.contains(e))
                                .map(|(e, w)| (*e, *w))
                                .unzip()
                        })
                        .collect();
                    {
                        let mut u: Vec<usize> = kept
                            .iter()
                            .flat_map(|k| k.0.iter().copied())
                            .filter(|e| ex.host_slot[*e].is_some())
                            .collect();
                        n_pairs += u.len();
                        u.sort_unstable();
                        u.dedup();
                        n_unique += u.len();
                        n_gpu += gpu_set.len();
                    }
                    let picks: Vec<(&[usize], &[f32])> =
                        kept.iter().map(|(a, b)| (&a[..], &b[..])).collect();
                    let inputs: Vec<&[f32]> = (0..t).map(|i| &xs[i * n..][..n]).collect();
                    let sums: Vec<f32> = ex.cpu_sums(&picks, &inputs, ff, limit).concat();
                    this.gpu.write_mapped(&this.cpu_t, 0, &sums)?;
                    cpu_s += tc.elapsed().as_secs_f64();
                    next = (2, &s.a, n as u32, this.cpu_t.buffer(), n as u32);
                }
            }
        }
        let c = bmix3(
            this,
            hp,
            &mut ops,
            t,
            cur,
            &this.blocks[0].attn,
            next,
            next3,
        );
        ops.push(Op {
            kernel: &this.e.fin,
            buffers: vec![&s.x[c], &this.out_norm, &this.hall],
            constants: vec![n as u32, f(hp.rms_eps)],
            groups: [1, 1, tu],
            independent: false,
        });
        mm(
            this,
            &mut ops,
            &this.output,
            &this.hall,
            0,
            n,
            &this.logits_t,
            0,
            hp.vocab,
            t,
            (0, 0),
        );
        ops.push(Op {
            kernel: &this.e.argmax,
            buffers: vec![&this.logits_t, this.token.buffer()],
            constants: vec![hp.vocab as u32, 0],
            groups: [1, 1, tu],
            independent: false,
        });
        this.mtp_follow(&mut ops, t, pos, Some(t - 1));
        let last = submit(this, &mut ops)?;
        for p in pending.drain(..) {
            p.wait()?;
        }
        last.wait()?;
        let mut tok = vec![0f32; t];
        this.gpu.read_mapped(&this.token, 0, &mut tok)?;
        if std::env::var_os("KUNO_GLM_TIMING").is_some() {
            eprintln!(
                "small batch of {t}: {:.1} ms; router waits {:.1} ms, CPU experts {:.1} ms ({n_unique} experts, {n_pairs} uses; {n_gpu} on the device)",
                started.elapsed().as_secs_f64() * 1e3,
                wait_s * 1e3,
                cpu_s * 1e3
            );
        }
        Ok(tok.iter().map(|v| v.to_bits()).collect())
    }

    /// The KDA blocks' states as after token `index` of the last small batch (its
    /// snapshots), and the NextN block's `last` as the main model's h there.
    fn restore(&self, index: usize) -> Result<()> {
        let hp = &self.model.hp;
        let d = hp.kda_head * hp.heads;
        let state = hp.heads * 128 * 128;
        let stride = state + 9 * d;
        let mut ops: Vec<Op> = Vec::new();
        for (il, blk) in self.blocks.iter().enumerate() {
            let (
                Attention::Kda {
                    hist, state: st, ..
                },
                Some(Some(snap)),
            ) = (&blk.attention, self.snap.get(il))
            else {
                continue;
            };
            for (dst, words, at) in [(st, state, 0), (hist, 9 * d, state)] {
                ops.push(Op {
                    kernel: &self.e.copyoff,
                    buffers: vec![snap, dst],
                    constants: vec![words as u32, (index * stride + at) as u32, 0],
                    groups: grid(words.div_ceil(256)),
                    independent: true,
                });
            }
        }
        if let Some(mtp) = &self.mtp {
            ops.push(Op {
                kernel: &self.e.copyoff,
                buffers: vec![&self.hall, &mtp.last],
                constants: vec![hp.embd as u32, (index * hp.embd) as u32, 0],
                groups: grid(hp.embd.div_ceil(256)),
                independent: false,
            });
        }
        if let Some(op) = ops.last_mut() {
            op.independent = false;
        }
        submit(self, &mut ops)?.wait()
    }

    /// Greedy decoding of `count` tokens after `prompt` (from a fresh sequence), drafting
    /// `drafts` tokens per step with the NextN block and keeping those the main model
    /// agrees with. `emit` gets each token as it is decided.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        count: usize,
        drafts: usize,
        emit: &mut dyn FnMut(u32),
    ) -> Result<SpecStats> {
        let mut stats = SpecStats::default();
        let drafts = if self.mtp.is_some() {
            drafts.min(self.snaps).min(SMALL - 1)
        } else {
            0
        };
        self.load_mtp()?;
        let mut next = self.prefill(prompt)?;
        // KUNO_GLM_PROFILE: the decode's kernels only.
        if let Ok(mut p) = self.profile.lock() {
            p.clear();
        }
        let mut made = 0;
        while made < count {
            emit(next);
            made += 1;
            if made >= count {
                break;
            }
            // The banks adapt as in `step`: a round of trades (from the verifications'
            // counts) copies while the drafts run; settled before the next verification.
            let round = self.trade_round()?;
            let t0 = std::time::Instant::now();
            let mut batch = vec![next];
            if drafts > 0 {
                // The drafts from the main model's h at the position before `next`.
                if let Some(mtp) = &self.mtp {
                    let mut ops: Vec<Op> = Vec::new();
                    ops.push(Op {
                        kernel: &self.e.copyoff,
                        buffers: vec![&mtp.last, &mtp.h],
                        constants: vec![self.model.hp.embd as u32, 0, 0],
                        groups: grid(self.model.hp.embd.div_ceil(256)),
                        independent: false,
                    });
                    submit(self, &mut ops)?.wait()?;
                }
                let mut x = next;
                for i in 0..drafts.min(count - made) {
                    x = self.mtp_draft(x, self.pos + i)?;
                    batch.push(x);
                }
            }
            stats.draft_s += t0.elapsed().as_secs_f64();
            if let Some((pass, trades)) = round {
                pass.wait()?;
                self.settle_round(&trades)?;
                stats.trades += trades.len();
            }
            let t1 = std::time::Instant::now();
            let start = self.pos;
            let z = if self.pipelines(batch.len()) {
                self.run_pipe(&batch, (batch.len() - 1) as u32)?
            } else {
                self.run_small(&batch, (batch.len() - 1) as u32)?
            };
            stats.verify_s += t1.elapsed().as_secs_f64();
            let mut accepted = 0;
            while accepted + 1 < batch.len() && batch[accepted + 1] == z[accepted] {
                accepted += 1;
            }
            stats.steps += 1;
            stats.drafted += batch.len() - 1;
            stats.accepted += accepted;
            // Kept: `next` and the accepted drafts; their choices are the output.
            for &tok in &z[..accepted] {
                if made < count {
                    emit(tok);
                    made += 1;
                }
            }
            if accepted + 1 < batch.len() {
                self.restore(accepted)?;
                self.pos = start + accepted + 1;
            }
            next = z[accepted];
        }
        Ok(stats)
    }
}

/// One token's `a + b + c` into `o` (b's stride 0 for zeros).
fn sum3_op<'a>(
    this: &'a Engine,
    a: &'a Buffer,
    b: &'a Buffer,
    b_st: u32,
    c: &'a Buffer,
    o: &'a Buffer,
) -> Op<'a> {
    let n = this.model.hp.embd;
    Op {
        kernel: &this.e.b.sum3,
        buffers: vec![a, b, c, o],
        constants: vec![n as u32, 1, n as u32, b_st, n as u32, n as u32],
        groups: grid(n.div_ceil(256)),
        independent: false,
    }
}

/// An mve.comp step over the batch's routing (`slots`: the source's slot per expert).
#[allow(clippy::too_many_arguments)]
fn mve_op<'a>(
    this: &'a Engine,
    w: &'a Buffer,
    sc: &'a Buffer,
    x: &'a Buffer,
    y: &'a Buffer,
    slots: &'a Buffer,
    constants: Vec<u32>,
    groups: [u32; 3],
) -> Op<'a> {
    Op {
        kernel: &this.e.mve,
        buffers: vec![w, sc, x, y, &this.s.routed, slots, &this.e.tables],
        constants,
        groups,
        independent: false,
    }
}

/// KUNO_GLM_TIMING: every MoE block's choices over a generation, counted (in memory
/// only): per block [expert] (uses, uses while on the device).
static USES: std::sync::Mutex<Vec<Vec<(u64, u64)>>> = std::sync::Mutex::new(Vec::new());

pub(super) fn note_uses(il: usize, e: usize, counts: &[u32], ex: &BlockExperts) {
    if std::env::var_os("KUNO_GLM_TIMING").is_none() {
        return;
    }
    let Ok(mut u) = USES.lock() else { return };
    if u.len() <= il {
        u.resize(il + 1, Vec::new());
    }
    if u[il].len() < e {
        u[il].resize(e, (0, 0));
    }
    for (k, &c) in counts.iter().enumerate().take(e) {
        u[il][k].0 += u64::from(c);
        if ex.host_slot[k].is_none() {
            u[il][k].1 += u64::from(c);
        }
    }
}

/// The share of the noted uses the device served, and the share the `slots` most used
/// experts of each block would have (one fixed bank for the whole generation).
pub fn uses_report(slots: usize) -> Option<(f64, f64)> {
    let mut u = USES.lock().ok()?;
    let (mut all, mut served, mut best) = (0u64, 0u64, 0u64);
    for b in u.iter() {
        let mut c: Vec<u64> = b.iter().map(|x| x.0).collect();
        all += c.iter().sum::<u64>();
        served += b.iter().map(|x| x.1).sum::<u64>();
        c.sort_unstable_by(|a, b| b.cmp(a));
        best += c.iter().take(slots).sum::<u64>();
    }
    u.clear();
    (all > 0).then(|| (served as f64 / all as f64, best as f64 / all as f64))
}
