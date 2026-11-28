//! Pipelined verification (KUNO_GLM_PIPE=1): a small batch's tokens each on their own lane
//! (`step`'s kernels, its scratch, router output and CPU sum), so the CPU runs one token's
//! routed experts while the device runs the next token's attention. A small batch shares
//! almost no routed experts between its tokens, so batching them saves the CPU nothing.
//!
//! Lanes are submitted in token order per block, so on the one queue token j's attention
//! at block il follows token j - 1's (its keys and the KDA state), and each submission
//! starts with a barrier on the earlier ones. Lane j is recorded from the block after its
//! last MoE block's router up to the next MoE block's router, shared expert and bank;
//! the host spins on the router's seq, hands the token's experts to the pool (several
//! tokens' at once), then records lane j's next run. With KUNO_GLM_GPU_SHARE 0 the results
//! equal `step`'s (the same kernels and CPU sums) for the same banks; a share runs on the
//! side queue between the lane's runs (timeline semaphores).
//!
//! On the B70 host (Ryzen 5900XT, dual-channel DDR4) this is slower than the batched
//! verification (~200 ms against ~170 for three tokens): the CPU's experts are bound by
//! the host's memory bandwidth (~26 GB/s for one token's three or four experts, ~33 for
//! large calls), which the overlap does not raise, and the device's copies share it.
use super::batch::{grid, mm, submit_sync};
use super::*;
use kuno_image::engine::gpu::Timeline;

/// A lane's place in the pass: its streams' buffer, the mixer's pending mode and outputs.
struct Run<'a> {
    cur: Option<usize>,
    mode: u32,
    o: &'a Buffer,
    o3: &'a Buffer,
    ops: Vec<Op<'a>>,
}

impl<'m> Engine<'m> {
    /// Whether small batches of `t` tokens can run pipelined.
    pub(super) fn pipelines(&self, t: usize) -> bool {
        t > 1 && t <= self.lanes.len() + 1
    }

    /// `run_small` pipelined (`pipelines(tokens.len())`): every position's greedy choice;
    /// KDA states after the first `snaps` tokens kept for `restore`; the NextN block follows.
    pub(super) fn run_pipe(&mut self, tokens: &[u32], snaps: u32) -> Result<Vec<u32>> {
        let hp = self.model.hp.clone();
        let (n, t) = (hp.embd, tokens.len());
        if !self.pipelines(t) {
            return Err("model_state_invalid");
        }
        if self.pos + t > self.max_positions {
            return Err("context_full");
        }
        self.load_all()?;
        let mut emb = vec![0f32; t * n];
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed
                .decode_row(tok as usize, &mut emb[i * n..][..n])?;
        }
        if let Some(mtp) = &self.mtp {
            self.gpu.write_mapped(&mtp.emb, 0, &emb)?;
        }
        let mut counts = vec![0u32; hp.layers * hp.experts];
        let out = self.pipe_pass(&emb, snaps, &mut counts)?;
        let last = (self.pos + t - 1) as u64;
        let e = hp.experts;
        for (il, ex) in self.experts.iter().enumerate() {
            if let Some(ex) = ex.get() {
                ex.add_batch(&counts[il * e..][..e], last, t);
                super::spec::note_uses(il, e, &counts[il * e..][..e], ex);
            }
        }
        self.pos += t;
        Ok(out)
    }

    fn pipe_pass(&self, emb: &[f32], snaps: u32, counts: &mut [u32]) -> Result<Vec<u32>> {
        let hp = &self.model.hp;
        let (n, used, e_count) = (hp.embd, hp.experts_used, hp.experts);
        let t = emb.len() / n;
        let pos = self.pos;
        let own = Lane {
            s: &self.s,
            host_x: &self.host_x,
            route: &self.route,
            cpu_out: &self.cpu_out,
        };
        let lanes: Vec<Lane> = std::iter::once(own)
            .chain(self.lanes.iter().map(|l| l.lane()))
            .take(t)
            .collect();
        for (j, lane) in lanes.iter().enumerate() {
            let row = &emb[j * n..][..n];
            let x4: Vec<f32> = (0..4).flat_map(|_| row.iter().copied()).collect();
            self.gpu.write_mapped(lane.host_x, 0, &x4)?;
        }
        let mut runs: Vec<Run> = lanes
            .iter()
            .map(|l| Run {
                cur: None,
                mode: 0,
                o: &l.s.out,
                o3: &l.s.zeros,
                ops: Vec::new(),
            })
            .collect();
        let started = std::time::Instant::now();
        let (mut cpu_s, mut rec_s) = (0f64, 0f64);
        // Submissions in flight, numbered; each lane's latest.
        let mut pending: std::collections::VecDeque<(u64, Passing)> = Default::default();
        let mut number = 0u64;
        let mut latest = vec![0u64; t];
        // Per lane: the MoE block whose router it waits for (Wait), whose experts the CPU
        // runs (Busy), or done.
        #[derive(Clone, Copy, PartialEq)]
        enum State {
            Wait(usize),
            Busy(usize),
            Done,
        }
        // With a side queue the device's share of the host experts runs there, beside the
        // main queue's attention: after the lane's run (main timeline: submission number
        // + 1), before its next (side timeline).
        let tls: Option<(Timeline, Timeline)> = if self.gpu.has_side() && self.stage.len() >= 2 {
            Some((self.gpu.timeline()?, self.gpu.timeline()?))
        } else {
            None
        };
        let mut side_v = 0u64;
        let mut side_wait: Vec<u64> = vec![0; t];
        let mut side_pending: Vec<Passing> = Vec::new();
        // KUNO_GLM_TIMING: per lane and MoE block, the times its run was submitted, its
        // router seen, its CPU sum started and ended (s since `started`).
        let mut ev: Vec<[f64; 5]> = Vec::new();
        let mut cur_ev: Vec<[f64; 5]> = vec![[0.0; 5]; t];
        let mut state = Vec::with_capacity(t);
        for (j, lane) in lanes.iter().enumerate() {
            cur_ev[j][0] = started.elapsed().as_secs_f64();
            let at = self.pipe_record(*lane, &mut runs[j], 0, pos + j, j, snaps)?;
            pending.push_back((
                number,
                main_submit(self, tls.as_ref(), &mut runs[j].ops, number, 0)?,
            ));
            latest[j] = number;
            number += 1;
            state.push(at.map_or(State::Done, State::Wait));
        }
        let at_word = ((2 * used + n) * 4) as u64;
        let share: f32 = std::env::var("KUNO_GLM_GPU_SHARE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.25);
        let (tx, rx) = std::sync::mpsc::channel::<(usize, Vec<f32>, f64, f64)>();
        let mut sums: Vec<Option<Vec<f32>>> = vec![None; t];
        let mut waited = std::time::Instant::now();
        let result: Result<()> = rayon::in_place_scope(|sc| {
            while state.iter().any(|s| *s != State::Done) {
                let mut moved = false;
                // Routers done: their tokens' experts to the pool.
                for j in 0..t {
                    let State::Wait(il) = state[j] else { continue };
                    let lane = lanes[j];
                    let mut word = [0f32; 1];
                    self.gpu.read_mapped(lane.route, at_word, &mut word)?;
                    if word[0].to_bits() != seq(pos + j, il) {
                        continue;
                    }
                    cur_ev[j][1] = started.elapsed().as_secs_f64();
                    // Everything submitted before lane j's latest is done (one queue).
                    while pending.front().is_some_and(|p| p.0 < latest[j]) {
                        if let Some((_, p)) = pending.pop_front() {
                            p.wait()?;
                        }
                    }
                    let mut r = vec![0f32; 2 * used + n];
                    self.gpu.read_mapped(lane.route, 0, &mut r)?;
                    let ids: Vec<usize> = r[..used].iter().map(|v| v.to_bits() as usize).collect();
                    for &id in &ids {
                        counts[il * e_count + id] += 1;
                    }
                    let ex = self.experts[il].get().ok_or("model_state_invalid")?;
                    // KUNO_GLM_GPU_SHARE of the token's host experts to the device (copied
                    // over the bus beside the CPU's reads), the rest to the CPU.
                    let host: Vec<usize> = ids
                        .iter()
                        .copied()
                        .filter(|e| ex.host_slot[*e].is_some())
                        .collect();
                    let take = match (&tls, self.lane_maps.get(j)) {
                        (Some(_), Some(_)) => {
                            ((host.len() as f32 * share).round() as usize).min(self.piece)
                        }
                        _ => 0,
                    };
                    let gpu_set: Vec<usize> = host[..take].to_vec();
                    if let (Some((main_tl, side_tl)), true) = (&tls, take > 0) {
                        let tr = std::time::Instant::now();
                        let stage = self.stage.last().ok_or("model_state_invalid")?;
                        let mut ops =
                            self.share_ops(lane, &self.lane_maps[j], stage, ex, il, &gpu_set)?;
                        side_v += 1;
                        side_pending.push(submit_sync(
                            self,
                            &mut ops,
                            true,
                            &[(main_tl, latest[j] + 1)],
                            &[(side_tl, side_v)],
                        )?);
                        side_wait[j] = side_v;
                        runs[j].o3 = &lane.s.gd;
                        rec_s += tr.elapsed().as_secs_f64();
                    }
                    let (keep, w): (Vec<usize>, Vec<f32>) = ids
                        .iter()
                        .zip(&r[used..2 * used])
                        .filter(|(e, _)| !gpu_set.contains(e))
                        .map(|(e, w)| (*e, *w))
                        .unzip();
                    let tx = tx.clone();
                    let (ff, limit) = (hp.ff_expert, hp.clamp_expert[il]);
                    sc.spawn(move |_| {
                        let t0 = started.elapsed().as_secs_f64();
                        let sum = ex.cpu_sum(&keep, &w, &r[2 * used..], ff, limit);
                        let _ = tx.send((j, sum, t0, started.elapsed().as_secs_f64()));
                    });
                    state[j] = State::Busy(il);
                    moved = true;
                }
                while let Ok((j, sum, a, b)) = rx.try_recv() {
                    sums[j] = Some(sum);
                    cpu_s += b - a;
                    cur_ev[j][2] = a;
                    cur_ev[j][3] = b;
                }
                // Sums done: each lane's next run, in token order per block (token j's
                // attention reads the keys and state token j - 1's wrote).
                for j in 0..t {
                    let State::Busy(il) = state[j] else { continue };
                    if j > 0 && matches!(state[j - 1], State::Busy(b) if b <= il) {
                        continue;
                    }
                    let Some(sum) = sums[j].take() else { continue };
                    let lane = lanes[j];
                    let tr = std::time::Instant::now();
                    self.gpu.write_mapped(lane.cpu_out, 0, &sum)?;
                    let at = self.pipe_record(lane, &mut runs[j], il + 1, pos + j, j, snaps)?;
                    let wait = std::mem::take(&mut side_wait[j]);
                    pending.push_back((
                        number,
                        main_submit(self, tls.as_ref(), &mut runs[j].ops, number, wait)?,
                    ));
                    latest[j] = number;
                    number += 1;
                    state[j] = at.map_or(State::Done, State::Wait);
                    rec_s += tr.elapsed().as_secs_f64();
                    cur_ev[j][4] = started.elapsed().as_secs_f64();
                    ev.push(cur_ev[j]);
                    cur_ev[j] = [cur_ev[j][4], 0.0, 0.0, 0.0, 0.0];
                    moved = true;
                }
                if moved {
                    waited = std::time::Instant::now();
                } else if waited.elapsed().as_secs() >= 2 {
                    // Nothing for 2 s: the device failed, or a submission is lost.
                    return Err("gpu_failed");
                } else {
                    std::hint::spin_loop();
                }
            }
            Ok(())
        });
        result?;
        for p in side_pending.drain(..) {
            p.wait()?;
        }
        let wait_s = started.elapsed().as_secs_f64() - rec_s;
        // The head on every position, then the NextN block's catching up.
        let mut ops: Vec<Op> = Vec::new();
        self.exact.store(true, std::sync::atomic::Ordering::Relaxed);
        mm(
            self,
            &mut ops,
            &self.output,
            &self.hall,
            0,
            n,
            &self.logits_t,
            0,
            hp.vocab,
            t,
            (0, 0),
        );
        ops.push(Op {
            kernel: &self.e.argmax,
            buffers: vec![&self.logits_t, self.token.buffer()],
            constants: vec![hp.vocab as u32, 0],
            groups: [1, 1, t as u32],
            independent: false,
        });
        self.mtp_follow(&mut ops, t, pos, Some(t - 1));
        let last = submit(self, &mut ops);
        self.exact
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let last = last?;
        for (_, p) in pending.drain(..) {
            p.wait()?;
        }
        last.wait()?;
        let mut tok = vec![0f32; t];
        self.gpu.read_mapped(&self.token, 0, &mut tok)?;
        if std::env::var_os("KUNO_GLM_TIMING").is_some() && !ev.is_empty() {
            let k = ev.len() as f64;
            let mean = |f: &dyn Fn(&[f64; 5]) -> f64| ev.iter().map(f).sum::<f64>() / k * 1e3;
            // CPU busy: the union of the sums' intervals.
            let mut iv: Vec<(f64, f64)> = ev.iter().map(|e| (e[2], e[3])).collect();
            iv.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (mut busy, mut end) = (0f64, 0f64);
            for (a, b) in iv {
                if b > end {
                    busy += b - a.max(end);
                    end = b;
                }
            }
            eprintln!(
                "  per lane-block: device {:.2} ms (submit to router), to the pool {:.2} ms, sum {:.2} ms, then submit {:.2} ms; CPU busy {:.1} ms",
                mean(&|e| e[1] - e[0]),
                mean(&|e| e[2] - e[1]),
                mean(&|e| e[3] - e[2]),
                mean(&|e| e[4] - e[3]),
                busy * 1e3
            );
        }
        if std::env::var_os("KUNO_GLM_TIMING").is_some() {
            eprintln!(
                "pipelined batch of {t}: {:.1} ms; CPU experts {:.1} ms (summed over tasks), recording {:.1} ms, the rest {:.1} ms",
                started.elapsed().as_secs_f64() * 1e3,
                cpu_s * 1e3,
                rec_s * 1e3,
                wait_s * 1e3
            );
        }
        Ok(tok.iter().map(|v| v.to_bits()).collect())
    }

    /// The device's share of lane's host experts at block `il` (`set`): copied whole into
    /// the side queue's staging buffer, their gate and up rows, the SwiGLU (every chosen expert's
    /// again: the bank's unchanged), their down rows added into `s.gd`.
    fn share_ops<'a>(
        &'a self,
        lane: Lane<'a>,
        map: &'a Mapped,
        stage: &'a Buffer,
        ex: &'a BlockExperts,
        il: usize,
        set: &[usize],
    ) -> Result<Vec<Op<'a>>> {
        let hp = &self.model.hp;
        let (n, ff, used) = (hp.embd, hp.ff_expert, hp.experts_used);
        let s = lane.s;
        let mut ops = Vec::new();
        let mut slots = vec![u32::MAX; hp.experts];
        let st = ex.stride.map(|v| v as u32);
        let rec: u32 = st.iter().sum();
        for (k, &e) in set.iter().enumerate() {
            slots[e] = k as u32;
            let (chunk, at) = ex.host_at(ex.host_slot[e].ok_or("model_state_invalid")?);
            let wide = rec.is_multiple_of(4) && at.is_multiple_of(4);
            let (kernel, q) = if wide {
                (&self.e.b.copy4, 4)
            } else {
                (&self.e.copyoff, 1)
            };
            ops.push(Op {
                kernel,
                buffers: vec![chunk.buffer(), stage],
                constants: vec![rec / q, at as u32 / q, k as u32 * rec / q],
                groups: grid((rec as usize / q as usize).div_ceil(256)),
                independent: true,
            });
        }
        if let Some(op) = ops.last_mut() {
            op.independent = false;
        }
        self.gpu
            .write_mapped(map, 0, &kuno_image::engine::gpu::floats(bytes_of(&slots)))?;
        let rpg = 128 / self.k.lanes;
        let (gk, dk) = (experts::kind(ex.gate_kind)?, experts::kind(ex.down_kind)?);
        let mve = |buffers: Vec<&'a Buffer>, constants: Vec<u32>, groups: [u32; 3]| Op {
            kernel: &self.e.mve,
            buffers,
            constants,
            groups,
            independent: false,
        };
        ops.push(mve(
            vec![
                stage,
                stage,
                &s.normed,
                &s.ge,
                &s.routed,
                map.buffer(),
                &self.e.tables,
            ],
            vec![
                0,
                gk,
                (2 * ff) as u32,
                n as u32,
                used as u32,
                0,
                rec,
                0,
                rec,
                st[0],
            ],
            [((2 * ff) as u32).div_ceil(rpg), used as u32, 1],
        ));
        ops.push(Op {
            kernel: &self.e.glu,
            buffers: vec![&s.ge, &s.gh],
            constants: vec![ff as u32, f(hp.clamp_expert[il]), used as u32],
            groups: [((used * ff) as u32).div_ceil(256), 1, 1],
            independent: false,
        });
        let mode = if ex.device.is_some() { 2 } else { 1 };
        ops.push(mve(
            vec![
                stage,
                stage,
                &s.gh,
                &s.gd,
                &s.routed,
                map.buffer(),
                &self.e.tables,
            ],
            vec![
                mode,
                dk,
                n as u32,
                ff as u32,
                used as u32,
                0,
                rec,
                st[0] + st[1],
                rec,
                st[0] + st[1] + st[2],
            ],
            [(n as u32).div_ceil(rpg), 1, 1],
        ));
        Ok(ops)
    }

    /// Records lane `j` (the token at `pos`) from block `from` through the next MoE block's
    /// router, shared expert and bank (returning that block), or to the end: the output
    /// norm into `hall` row j. KDA blocks' states go to snapshot j when j < `snaps`.
    fn pipe_record<'a>(
        &'a self,
        lane: Lane<'a>,
        run: &mut Run<'a>,
        from: usize,
        pos: usize,
        j: usize,
        snaps: u32,
    ) -> Result<Option<usize>> {
        let hp = &self.model.hp;
        let n = hp.embd;
        let s = lane.s;
        for il in from..self.blocks.len() {
            let blk = &self.blocks[il];
            let c = mix(
                self,
                lane,
                hp,
                &mut run.ops,
                run.o3,
                run.cur,
                &blk.attn,
                run.mode,
                run.o,
            );
            run.cur = Some(c);
            one_attention(self, lane, &mut run.ops, &blk.attention, pos);
            if let (Attention::Kda { hist, state, .. }, Some(Some(snap)), true) =
                (&blk.attention, self.snap.get(il), (j as u32) < snaps)
            {
                let d = hp.kda_head * hp.heads;
                let words = hp.heads * 128 * 128;
                let stride = words + 9 * d;
                for (src, len, at) in [(state, words, 0), (hist, 9 * d, words)] {
                    run.ops.push(Op {
                        kernel: &self.e.copyoff,
                        buffers: vec![src, snap],
                        constants: vec![len as u32, 0, (j * stride + at) as u32],
                        groups: grid(len.div_ceil(256)),
                        independent: true,
                    });
                }
            }
            let c = mix(
                self,
                lane,
                hp,
                &mut run.ops,
                run.o3,
                run.cur,
                &blk.ffn_mixer,
                1,
                &s.a,
            );
            run.cur = Some(c);
            match &blk.ffn {
                Ffn::Dense { gu, down } => {
                    (run.mode, run.o) = one_dense(self, lane, &mut run.ops, il, gu, down);
                }
                Ffn::Moe {
                    router,
                    bias,
                    gu,
                    down,
                } => {
                    one_router(self, lane, &mut run.ops, il, router, bias, pos);
                    run.o3 = one_back(self, lane, &mut run.ops, il, gu, down)?;
                    run.mode = 2;
                    run.o = &s.a;
                    return Ok(Some(il));
                }
            }
        }
        // The last block's FFN mixer (with any block's product, unused), the norm.
        let c = mix(
            self,
            lane,
            hp,
            &mut run.ops,
            run.o3,
            run.cur,
            &self.blocks[0].attn,
            run.mode,
            run.o,
        );
        run.ops.push(Op {
            kernel: &self.e.fin,
            buffers: vec![&s.x[c], &self.out_norm, &s.normed],
            constants: vec![n as u32, f(hp.rms_eps)],
            groups: [1, 1, 1],
            independent: false,
        });
        run.ops.push(Op {
            kernel: &self.e.copyoff,
            buffers: vec![&s.normed, &self.hall],
            constants: vec![n as u32, 0, (j * n) as u32],
            groups: grid(n.div_ceil(256)),
            independent: false,
        });
        Ok(None)
    }

    /// The CPU's expert sums alone (KUNO_GLM_SUMBENCH): `per` host experts per call over
    /// every MoE block, as a token's; returns GB/s.
    pub fn bench_cpu_sums(&self, per: usize, rounds: usize) -> f64 {
        let hp = &self.model.hp;
        let n = hp.embd;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 7919) % 255) as f32 / 37.0 - 3.4)
            .collect();
        let mut seed = 0x9e3779b97f4a7c15u64;
        let (mut bytes, mut secs) = (0f64, 0f64);
        for _ in 0..rounds {
            for (il, ex) in self.experts.iter().enumerate() {
                let Some(ex) = ex.get() else { continue };
                let host: Vec<usize> = (0..hp.experts)
                    .filter(|e| ex.host_slot[*e].is_some())
                    .collect();
                if host.len() < per {
                    continue;
                }
                let ids: Vec<usize> = (0..per)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        host[seed as usize % host.len()]
                    })
                    .collect();
                let w = vec![0.1f32; per];
                let t = std::time::Instant::now();
                let s = ex.cpu_sum(&ids, &w, &x, hp.ff_expert, hp.clamp_expert[il]);
                std::hint::black_box(s);
                secs += t.elapsed().as_secs_f64();
                bytes += (per * ex.stride.iter().sum::<usize>() * 4) as f64;
            }
        }
        bytes / secs / 1e9
    }
}

/// A main-queue submission of the pipeline: with timelines (a side queue) signalling the
/// main one at `number + 1`, after the side one reaches `wait` (when not 0).
fn main_submit<'a>(
    this: &'a Engine,
    tls: Option<&(Timeline, Timeline)>,
    ops: &mut Vec<Op<'a>>,
    number: u64,
    wait: u64,
) -> Result<Passing> {
    match tls {
        Some((main_tl, side_tl)) => {
            let waits: Vec<(&Timeline, u64)> = if wait > 0 {
                vec![(side_tl, wait)]
            } else {
                Vec::new()
            };
            submit_sync(this, ops, false, &waits, &[(main_tl, number + 1)])
        }
        None => submit(this, ops),
    }
}
