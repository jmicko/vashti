//! The exact CPU forward pass of GLM 5.3 Flash, one token at a time (f32 values, f64 sums in
//! the products), following llama.cpp's `glm5next` graph. Packed weights stay in memory
//! (experts are read from the file when routed to); rows are decoded as they are used. It
//! attends to every earlier position, which is the model's attention while the context is at
//! most `indexer_top_k + indexer_kpool - 1` tokens (2051), where the indexer selects all.
use rayon::prelude::*;

use crate::{Hparams, Matrix, Model, Result};

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
/// `x / sqrt(mean(x^2) + eps)`, times `weight` if given.
fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64;
    let scale = (1.0 / (mean + f64::from(eps)).sqrt()) as f32;
    match weight {
        Some(w) => x.iter().zip(w).map(|(v, w)| v * scale * w).collect(),
        None => x.iter().map(|v| v * scale).collect(),
    }
}
/// ggml's `l2_norm`: `x / max(|x|, eps)`.
fn l2_norm(x: &mut [f32], eps: f32) {
    let sum = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>();
    let scale = (1.0 / sum.sqrt().max(f64::from(eps))) as f32;
    x.iter_mut().for_each(|v| *v *= scale);
}
/// ggml's clamped SwiGLU: `silu(min(gate, limit)) * clamp(up, -limit, limit)`.
fn swiglu(gate: &[f32], up: &[f32], limit: f32) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| {
            if limit > 1e-6 {
                silu(g.min(limit)) * u.clamp(-limit, limit)
            } else {
                silu(*g) * u
            }
        })
        .collect()
}

struct Mixer {
    function: Matrix,
    base: Vec<f32>,
    scale: Vec<f32>,
}
#[allow(clippy::large_enum_variant)]
enum Attention {
    Kda {
        q: Matrix,
        k: Matrix,
        v: Matrix,
        /// `[channel][conv]` for q, k, v concatenated.
        conv: Vec<f32>,
        f_a: Matrix,
        f_b: Matrix,
        g_a: Matrix,
        g_b: Matrix,
        beta: Matrix,
        /// `-exp(A_log)` per head.
        a: Vec<f32>,
        dt_bias: Vec<f32>,
        out_norm: Vec<f32>,
        out: Matrix,
    },
    Dsa {
        q_a: Matrix,
        q_a_norm: Vec<f32>,
        q_b: Matrix,
        kv_a: Matrix,
        kv_a_norm: Vec<f32>,
        /// `[head][latent]` rows of `qk_head`.
        k_b: Matrix,
        /// `[head][v_head]` rows of `kv_lora`.
        v_b: Matrix,
        out: Matrix,
    },
}
enum Ffn {
    Dense {
        gate: Matrix,
        up: Matrix,
        down: Matrix,
    },
    Moe {
        router: Matrix,
        bias: Vec<f32>,
        gate: Matrix,
        up: Matrix,
        down: Matrix,
    },
}
struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    hc_attn: Mixer,
    hc_ffn: Mixer,
    attention: Attention,
    ffn: Ffn,
}

/// One block's state across tokens.
enum State {
    Kda {
        /// The last `conv - 1` q|k|v projections, oldest first.
        conv: Vec<Vec<f32>>,
        /// `[head][key][value]`.
        s: Vec<f32>,
    },
    Dsa {
        /// Each earlier position's normed latent.
        latents: Vec<Vec<f32>>,
    },
}

pub struct Reference<'m> {
    model: &'m Model,
    layers: Vec<Layer>,
    embed: Matrix,
    output: Matrix,
    out_norm: Vec<f32>,
    states: Vec<State>,
    /// Per block, after it: the mean of the four streams of the last token (comparisons).
    pub trace: Vec<Vec<f32>>,
}

impl<'m> Reference<'m> {
    pub fn new(model: &'m Model) -> Result<Self> {
        let hp = &model.hp;
        let m = |n: &str| model.matrix(n);
        let v = |n: &str| model.vector(n);
        let mut layers = Vec::with_capacity(hp.layers);
        for il in 0..hp.layers {
            let b = |n: &str| format!("blk.{il}.{n}");
            let mixer = |part: &str| -> Result<Mixer> {
                Ok(Mixer {
                    function: m(&b(&format!("hc_{part}_fn.weight")))?,
                    base: v(&b(&format!("hc_{part}_base.weight")))?,
                    scale: v(&b(&format!("hc_{part}_scale.weight")))?,
                })
            };
            let attention = if hp.dsa[il] {
                Attention::Dsa {
                    q_a: m(&b("attn_q_a.weight"))?,
                    q_a_norm: v(&b("attn_q_a_norm.weight"))?,
                    q_b: m(&b("attn_q_b.weight"))?,
                    kv_a: m(&b("attn_kv_a_mqa.weight"))?,
                    kv_a_norm: v(&b("attn_kv_a_norm.weight"))?,
                    k_b: m(&b("attn_k_b.weight"))?,
                    v_b: m(&b("attn_v_b.weight"))?,
                    out: m(&b("attn_output.weight"))?,
                }
            } else {
                let mut conv = Vec::new();
                for part in ["q", "k", "v"] {
                    conv.extend(v(&b(&format!("ssm_conv1d_{part}.weight")))?);
                }
                Attention::Kda {
                    q: m(&b("attn_q.weight"))?,
                    k: m(&b("attn_k.weight"))?,
                    v: m(&b("attn_v.weight"))?,
                    conv,
                    f_a: m(&b("ssm_f_a.weight"))?,
                    f_b: m(&b("ssm_f_b.weight"))?,
                    g_a: m(&b("ssm_g_a.weight"))?,
                    g_b: m(&b("ssm_g_b.weight"))?,
                    beta: m(&b("ssm_beta.weight"))?,
                    a: v(&b("ssm_a"))?,
                    dt_bias: v(&b("ssm_dt.bias"))?,
                    out_norm: v(&b("ssm_norm.weight"))?,
                    out: m(&b("attn_output.weight"))?,
                }
            };
            let ffn = if il < hp.dense_lead {
                Ffn::Dense {
                    gate: m(&b("ffn_gate.weight"))?,
                    up: m(&b("ffn_up.weight"))?,
                    down: m(&b("ffn_down.weight"))?,
                }
            } else {
                Ffn::Moe {
                    router: m(&b("ffn_gate_inp.weight"))?,
                    bias: v(&b("exp_probs_b.bias"))?,
                    gate: m(&b("ffn_gate_shexp.weight"))?,
                    up: m(&b("ffn_up_shexp.weight"))?,
                    down: m(&b("ffn_down_shexp.weight"))?,
                }
            };
            layers.push(Layer {
                attn_norm: v(&b("attn_norm.weight"))?,
                ffn_norm: v(&b("ffn_norm.weight"))?,
                hc_attn: mixer("attn")?,
                hc_ffn: mixer("ffn")?,
                attention,
                ffn,
            });
        }
        let states = (0..hp.layers)
            .map(|il| {
                if hp.dsa[il] {
                    State::Dsa {
                        latents: Vec::new(),
                    }
                } else {
                    let d = hp.kda_head * hp.heads;
                    State::Kda {
                        conv: vec![vec![0.0; 3 * d]; hp.conv - 1],
                        s: vec![0.0; hp.heads * hp.kda_head * hp.kda_head],
                    }
                }
            })
            .collect();
        Ok(Self {
            model,
            layers,
            embed: m("token_embd.weight")?,
            output: m("output.weight")?,
            out_norm: v("output_norm.weight")?,
            states,
            trace: Vec::new(),
        })
    }

    /// Runs one token; returns the next-token logits when `logits` is set.
    pub fn step(&mut self, token: u32, logits: bool) -> Result<Option<Vec<f32>>> {
        let hp = self.model.hp.clone();
        let (n, hc) = (hp.embd, hp.hc);
        let mut row = vec![0f32; n];
        self.embed.decode_row(token as usize, &mut row)?;
        // Four exact copies of the embedding.
        let mut x: Vec<f32> = (0..hc).flat_map(|_| row.iter().copied()).collect();
        self.trace.clear();
        for il in 0..hp.layers {
            let (input, post, comb) = mix_pre(&self.layers[il].hc_attn, &x, &hp);
            let normed = rms_norm(&input, Some(&self.layers[il].attn_norm), hp.rms_eps);
            let out = self.attention(il, &normed)?;
            x = mix_post(&out, &x, &post, &comb, hc);
            let (input, post, comb) = mix_pre(&self.layers[il].hc_ffn, &x, &hp);
            let normed = rms_norm(&input, Some(&self.layers[il].ffn_norm), hp.rms_eps);
            let out = self.ffn(il, &normed)?;
            x = mix_post(&out, &x, &post, &comb, hc);
            self.trace.push(stream_mean(&x, n, hc));
        }
        if !logits {
            return Ok(None);
        }
        let mean = stream_mean(&x, n, hc);
        let normed = rms_norm(&mean, Some(&self.out_norm), hp.rms_eps);
        Ok(Some(self.output.matvec(&normed)))
    }

    fn attention(&mut self, il: usize, x: &[f32]) -> Result<Vec<f32>> {
        let hp = &self.model.hp;
        let (heads, eps) = (hp.heads, hp.rms_eps);
        match (&self.layers[il].attention, &mut self.states[il]) {
            (
                Attention::Kda {
                    q,
                    k,
                    v,
                    conv,
                    f_a,
                    f_b,
                    g_a,
                    g_b,
                    beta,
                    a,
                    dt_bias,
                    out_norm,
                    out,
                },
                State::Kda { conv: history, s },
            ) => {
                let dh = hp.kda_head;
                let d = dh * heads;
                let cw = hp.conv;
                let mut qkv = q.matvec(x);
                qkv.extend(k.matvec(x));
                qkv.extend(v.matvec(x));
                // Causal conv over the last `cw` projections, then SiLU.
                let mut c = vec![0f32; 3 * d];
                for (ch, out) in c.iter_mut().enumerate() {
                    let w = &conv[ch * cw..][..cw];
                    let mut sum = 0f32;
                    for (j, wj) in w.iter().enumerate() {
                        let value = if j + 1 == cw { qkv[ch] } else { history[j][ch] };
                        sum += wj * value;
                    }
                    *out = silu(sum);
                }
                history.remove(0);
                history.push(qkv);
                let (mut qh, rest) = c.split_at_mut(d);
                let (mut kh, vh) = rest.split_at_mut(d);
                for h in 0..heads {
                    l2_norm(&mut qh[h * dh..][..dh], 1e-6);
                    l2_norm(&mut kh[h * dh..][..dh], 1e-6);
                }
                let (qh, kh) = (&mut qh, &mut kh);
                let mut g = f_b.matvec(&f_a.matvec(x));
                for (i, gi) in g.iter_mut().enumerate() {
                    let t = (*gi + dt_bias[i]) * a[i / dh];
                    *gi = sigmoid(-t) * hp.gate_lower_bound;
                }
                let betas: Vec<f32> = beta.matvec(x).into_iter().map(sigmoid).collect();
                let scale = 1.0 / (dh as f32).sqrt();
                let mut o = vec![0f32; d];
                s.par_chunks_exact_mut(dh * dh)
                    .zip(o.par_chunks_exact_mut(dh))
                    .enumerate()
                    .for_each(|(h, (s, o))| {
                        let (q, k, v) = (
                            &qh[h * dh..][..dh],
                            &kh[h * dh..][..dh],
                            &vh[h * dh..][..dh],
                        );
                        let g = &g[h * dh..][..dh];
                        // s[key][value]: decay each key row, the delta rule, read out with q.
                        for (i, row) in s.chunks_exact_mut(dh).enumerate() {
                            let decay = g[i].exp();
                            row.iter_mut().for_each(|e| *e *= decay);
                        }
                        let mut predicted = vec![0f32; dh];
                        for (i, row) in s.chunks_exact(dh).enumerate() {
                            for (p, e) in predicted.iter_mut().zip(row) {
                                *p += e * k[i];
                            }
                        }
                        let delta: Vec<f32> = v
                            .iter()
                            .zip(&predicted)
                            .map(|(v, p)| (v - p) * betas[h])
                            .collect();
                        for (i, row) in s.chunks_exact_mut(dh).enumerate() {
                            for (e, dj) in row.iter_mut().zip(&delta) {
                                *e += k[i] * dj;
                            }
                        }
                        for (i, row) in s.chunks_exact(dh).enumerate() {
                            let qi = q[i] * scale;
                            for (oj, e) in o.iter_mut().zip(row) {
                                *oj += e * qi;
                            }
                        }
                    });
                let gate = g_b.matvec(&g_a.matvec(x));
                let mut gated = vec![0f32; d];
                for h in 0..heads {
                    let normed = rms_norm(&o[h * dh..][..dh], Some(out_norm), eps);
                    for j in 0..dh {
                        gated[h * dh + j] = normed[j] * sigmoid(gate[h * dh + j]);
                    }
                }
                Ok(out.matvec(&gated))
            }
            (
                Attention::Dsa {
                    q_a,
                    q_a_norm,
                    q_b,
                    kv_a,
                    kv_a_norm,
                    k_b,
                    v_b,
                    out,
                },
                State::Dsa { latents },
            ) => {
                let (qk, vd, lora) = (hp.qk_head, hp.v_head, hp.kv_lora);
                let qr = rms_norm(&q_a.matvec(x), Some(q_a_norm), eps);
                let q = q_b.matvec(&qr);
                let latent = rms_norm(&kv_a.matvec(x), Some(kv_a_norm), eps);
                latents.push(latent);
                let scale = 1.0 / (qk as f32).sqrt();
                let kb = k_b.decode_all()?;
                let vb = v_b.decode_all()?;
                let heads_out: Vec<Vec<f32>> = (0..heads)
                    .into_par_iter()
                    .map(|h| {
                        let qh = &q[h * qk..][..qk];
                        // The query through the key's up-projection: a latent-width query.
                        let absorbed: Vec<f32> = (0..lora)
                            .map(|c| {
                                let w = &kb[(h * lora + c) * qk..][..qk];
                                w.iter().zip(qh).map(|(a, b)| f64::from(a * b)).sum::<f64>() as f32
                            })
                            .collect();
                        let scores: Vec<f64> = latents
                            .iter()
                            .map(|l| {
                                absorbed
                                    .iter()
                                    .zip(l)
                                    .map(|(a, b)| f64::from(a * b))
                                    .sum::<f64>()
                                    * f64::from(scale)
                            })
                            .collect();
                        let top = scores.iter().copied().fold(f64::MIN, f64::max);
                        let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                        let total: f64 = weights.iter().sum();
                        let mut mixed = vec![0f64; lora];
                        for (w, l) in weights.iter().zip(latents.iter()) {
                            for (m, v) in mixed.iter_mut().zip(l) {
                                *m += w / total * f64::from(*v);
                            }
                        }
                        (0..vd)
                            .map(|j| {
                                let w = &vb[(h * vd + j) * lora..][..lora];
                                w.iter()
                                    .zip(&mixed)
                                    .map(|(a, b)| f64::from(*a) * b)
                                    .sum::<f64>() as f32
                            })
                            .collect()
                    })
                    .collect();
                Ok(out.matvec(&heads_out.concat()))
            }
            _ => Err("model_state_invalid"),
        }
    }

    fn ffn(&self, il: usize, x: &[f32]) -> Result<Vec<f32>> {
        let hp = &self.model.hp;
        match &self.layers[il].ffn {
            Ffn::Dense { gate, up, down } => {
                let h = swiglu(&gate.matvec(x), &up.matvec(x), hp.clamp_shared[il]);
                Ok(down.matvec(&h))
            }
            Ffn::Moe {
                router,
                bias,
                gate,
                up,
                down,
            } => {
                let probs: Vec<f32> = router.matvec(x).into_iter().map(sigmoid).collect();
                // The bias chooses the experts; the weights are the unbiased scores.
                let mut order: Vec<usize> = (0..hp.experts).collect();
                order.sort_by(|a, b| {
                    (probs[*b] + bias[*b])
                        .partial_cmp(&(probs[*a] + bias[*a]))
                        .unwrap()
                        .then(a.cmp(b))
                });
                let chosen = &order[..hp.experts_used];
                let mut weights: Vec<f32> = chosen.iter().map(|e| probs[*e]).collect();
                if hp.expert_norm {
                    let sum = weights.iter().sum::<f32>().max(6.103_515_6e-5);
                    weights.iter_mut().for_each(|w| *w /= sum);
                }
                weights.iter_mut().for_each(|w| *w *= hp.expert_scale);
                let m = self.model;
                let f = hp.ff_expert;
                let b = |n: &str| format!("blk.{il}.{n}");
                let mut out = vec![0f32; hp.embd];
                for (e, w) in chosen.iter().zip(&weights) {
                    let g = m.rows(&b("ffn_gate_exps.weight"), e * f, f)?.matvec(x);
                    let u = m.rows(&b("ffn_up_exps.weight"), e * f, f)?.matvec(x);
                    let h = swiglu(&g, &u, hp.clamp_expert[il]);
                    let d = m
                        .rows(&b("ffn_down_exps.weight"), e * hp.embd, hp.embd)?
                        .matvec(&h);
                    for (o, v) in out.iter_mut().zip(&d) {
                        *o += w * v;
                    }
                }
                let h = swiglu(&gate.matvec(x), &up.matvec(x), hp.clamp_shared[il]);
                for (o, v) in out.iter_mut().zip(down.matvec(&h)) {
                    *o += v;
                }
                Ok(out)
            }
        }
    }
}

fn stream_mean(x: &[f32], n: usize, hc: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0..hc).map(|h| x[h * n + i]).sum::<f32>() / hc as f32)
        .collect()
}

/// The hyper-connection mixer before a sublayer: its input (the streams weighted by `pre`),
/// and `post` (per stream) and `comb` (`[src][dst]`) for `mix_post`.
fn mix_pre(m: &Mixer, x: &[f32], hp: &Hparams) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (n, hc, eps) = (hp.embd, hp.hc, hp.hc_eps);
    let mixes = m.function.matvec(&rms_norm(x, None, hp.rms_eps));
    let pre: Vec<f32> = (0..hc)
        .map(|h| sigmoid(mixes[h] * m.scale[0] + m.base[h]) + eps)
        .collect();
    let post: Vec<f32> = (0..hc)
        .map(|h| 2.0 * sigmoid(mixes[hc + h] * m.scale[1] + m.base[hc + h]))
        .collect();
    // comb[src][dst]: a softmax over dst, then Sinkhorn normalization of columns and rows.
    let mut comb: Vec<f32> = (0..hc * hc)
        .map(|i| mixes[2 * hc + i] * m.scale[2] + m.base[2 * hc + i])
        .collect();
    for row in comb.chunks_exact_mut(hc) {
        let top = row.iter().copied().fold(f32::MIN, f32::max);
        let sum: f32 = row.iter().map(|v| (v - top).exp()).sum();
        row.iter_mut()
            .for_each(|v| *v = (*v - top).exp() / sum + eps);
    }
    let cols = |c: &mut Vec<f32>| {
        for dst in 0..hc {
            let sum: f32 = (0..hc).map(|src| c[src * hc + dst]).sum::<f32>() + eps;
            (0..hc).for_each(|src| c[src * hc + dst] /= sum);
        }
    };
    let rows = |c: &mut Vec<f32>| {
        for row in c.chunks_exact_mut(hc) {
            let sum = row.iter().sum::<f32>() + eps;
            row.iter_mut().for_each(|v| *v /= sum);
        }
    };
    cols(&mut comb);
    for _ in 1..hp.sinkhorn {
        rows(&mut comb);
        cols(&mut comb);
    }
    let input = (0..n)
        .map(|i| (0..hc).map(|h| x[h * n + i] * pre[h]).sum())
        .collect();
    (input, post, comb)
}

/// The mixer after a sublayer: stream `dst` is `out * post[dst] + sum_src x[src] comb[src][dst]`.
fn mix_post(out: &[f32], x: &[f32], post: &[f32], comb: &[f32], hc: usize) -> Vec<f32> {
    let n = out.len();
    let mut next = vec![0f32; hc * n];
    for dst in 0..hc {
        for i in 0..n {
            let mut v = out[i] * post[dst];
            for src in 0..hc {
                v += x[src * n + i] * comb[src * hc + dst];
            }
            next[dst * n + i] = v;
        }
    }
    next
}
