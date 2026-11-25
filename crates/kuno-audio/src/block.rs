//! The LFM block shared by the backbone, the audio encoder, the depth decoder and the codec:
//! a pre-norm operator (gated short convolution, or GQA attention with q/k RMS norms, NEOX
//! rotary, an optional per-head sigmoid gate and an optional sliding window), an optional
//! post-operator norm, then a pre-norm SwiGLU with an optional post norm. Each block keeps
//! its own state across calls (the convolution's last inputs, the attention's keys and values).
use rayon::prelude::*;

use crate::tensor::{Linear, rms_norm, rope_neox, sigmoid, silu, softmax};

pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub gate: Option<Linear>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub factors: Option<Vec<f32>>,
    /// Keys within this many positions (the query's own included); `None`: all.
    pub window: Option<usize>,
}
pub struct Conv {
    /// `[b | c | x]` of the input.
    pub in_proj: Linear,
    /// `[channel][tap]`, the last tap at the current position.
    pub taps: Vec<f32>,
    pub kernel: usize,
    pub out_proj: Linear,
}
#[allow(clippy::large_enum_variant)]
pub enum Op {
    Attention(Attention),
    Conv(Conv),
}
pub struct Block {
    pub width: usize,
    pub eps: f32,
    pub ln1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub post_op: Option<Vec<f32>>,
    pub post_ffn: Option<Vec<f32>>,
    pub op: Op,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// A block's state: the convolution's last `kernel - 1` inputs, or the attention's keys and
/// values (`kv_heads * head_dim` per position) and the position of the next token.
#[derive(Clone, Default)]
pub struct State {
    pub history: Vec<Vec<f32>>,
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    pub next: usize,
}

impl Block {
    /// Runs `x` (`n` rows of `width`) through the block; `causal` attention sees earlier and
    /// cached positions only, otherwise every row of `x` (no cache).
    pub fn forward(&self, x: &mut [f32], state: &mut State, causal: bool) {
        let w = self.width;
        let mut h = x.to_vec();
        rms_norm(&mut h, Some(&self.ln1), w, self.eps);
        let mut out = match &self.op {
            Op::Conv(c) => conv(c, &h, w, state),
            Op::Attention(a) => attention(a, &h, w, state, causal, self.eps),
        };
        if let Some(n) = &self.post_op {
            rms_norm(&mut out, Some(n), w, self.eps);
        }
        x.iter_mut().zip(&out).for_each(|(a, b)| *a += b);
        let mut h = x.to_vec();
        rms_norm(&mut h, Some(&self.ln2), w, self.eps);
        let g = self.gate.forward(&h);
        let u = self.up.forward(&h);
        let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
        let mut f = self.down.forward(&act);
        if let Some(n) = &self.post_ffn {
            rms_norm(&mut f, Some(n), w, self.eps);
        }
        x.iter_mut().zip(&f).for_each(|(a, b)| *a += b);
    }
}

fn conv(c: &Conv, h: &[f32], w: usize, state: &mut State) -> Vec<f32> {
    let n = h.len() / w;
    let bcx = c.in_proj.forward(h);
    if state.history.is_empty() {
        state.history = vec![vec![0.0; w]; c.kernel - 1];
    }
    let mut y = vec![0f32; n * w];
    for t in 0..n {
        let row = &bcx[t * 3 * w..][..3 * w];
        let (b, cc, xx) = (&row[..w], &row[w..2 * w], &row[2 * w..]);
        let bx: Vec<f32> = b.iter().zip(xx).map(|(a, b)| a * b).collect();
        for ch in 0..w {
            let taps = &c.taps[ch * c.kernel..][..c.kernel];
            let mut sum = taps[c.kernel - 1] * bx[ch];
            for (j, hist) in state.history.iter().enumerate() {
                sum += taps[j] * hist[ch];
            }
            y[t * w + ch] = cc[ch] * sum;
        }
        state.history.remove(0);
        state.history.push(bx);
    }
    c.out_proj.forward(&y)
}

fn attention(a: &Attention, h: &[f32], w: usize, state: &mut State, causal: bool, eps: f32) -> Vec<f32> {
    let n = h.len() / w;
    let (hd, heads, kvh) = (a.head_dim, a.heads, a.kv_heads);
    let mut q = a.q.forward(h);
    let mut k = a.k.forward(h);
    let v = a.v.forward(h);
    rms_norm(&mut q, Some(&a.q_norm), hd, eps);
    rms_norm(&mut k, Some(&a.k_norm), hd, eps);
    let start = if causal { state.next } else { 0 };
    for t in 0..n {
        for head in q[t * heads * hd..][..heads * hd].chunks_exact_mut(hd) {
            rope_neox(head, start + t, a.rope_base, a.factors.as_deref());
        }
        for head in k[t * kvh * hd..][..kvh * hd].chunks_exact_mut(hd) {
            rope_neox(head, start + t, a.rope_base, a.factors.as_deref());
        }
    }
    // Keys and values visible to this call: the cache, then this call's rows.
    let (mut keys, mut values) = if causal {
        (std::mem::take(&mut state.keys), std::mem::take(&mut state.values))
    } else {
        (Vec::new(), Vec::new())
    };
    let base = start - keys.len().min(start);
    let cached = keys.len();
    for t in 0..n {
        keys.push(k[t * kvh * hd..][..kvh * hd].to_vec());
        values.push(v[t * kvh * hd..][..kvh * hd].to_vec());
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let mixed: Vec<f32> = (0..n * heads)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (t, head) = (i / heads, i % heads);
            let g = head / (heads / kvh);
            let qh = &q[(t * heads + head) * hd..][..hd];
            let pos = start + t;
            // Key j sits at position base + j.
            let last = if causal { cached + t } else { keys.len() - 1 };
            let first = match (causal, a.window) {
                (true, Some(win)) => (pos + 1).saturating_sub(win).max(base) - base,
                _ => 0,
            };
            let mut scores: Vec<f32> = (first..=last)
                .map(|j| crate::tensor::dot(qh, &keys[j][g * hd..][..hd]) * scale)
                .collect();
            softmax(&mut scores);
            let mut o = vec![0f32; hd];
            for (p, j) in scores.iter().zip(first..=last) {
                for (oi, vi) in o.iter_mut().zip(&values[j][g * hd..][..hd]) {
                    *oi += p * vi;
                }
            }
            o
        })
        .collect();
    let mut mixed = mixed;
    if let Some(gate) = &a.gate {
        let gates = gate.forward(h);
        for t in 0..n {
            for head in 0..heads {
                let s = sigmoid(gates[t * heads + head]);
                mixed[(t * heads + head) * hd..][..hd].iter_mut().for_each(|v| *v *= s);
            }
        }
    }
    if causal {
        // Keep what a later call can still see.
        let keep = a.window.map_or(keys.len(), |win| win.saturating_sub(1).min(keys.len()));
        let drop = keys.len() - keep;
        keys.drain(..drop);
        values.drain(..drop);
        state.keys = keys;
        state.values = values;
        state.next = start + n;
    }
    a.o.forward(&mixed)
}
