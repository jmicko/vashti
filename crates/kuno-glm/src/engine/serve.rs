//! Serving: sampling the next token on the host, a saved point of the sequence (so the next
//! request of a conversation, which repeats this one's prompt, reads only what is new), and
//! the logits the host samples from. Nothing here keeps content beyond the engine's own
//! state (positions and recurrent states in device memory, as any decode does).
use super::batch::grid;
use super::*;

/// Ollama's sampling (top_k, temperature, softmax, top_p, min_p, then a draw), from a seeded
/// generator: the same seed and logits give the same tokens. Temperature 0 is greedy.
pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    state: u64,
    scratch: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(seed: u64, temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p,
            state: seed,
            scratch: Vec::new(),
        }
    }

    /// Greedy (the device's own choice can stand in).
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// splitmix64, as a uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn choose(&mut self, logits: &[f32]) -> u32 {
        let argmax = || {
            let mut best = 0;
            for (i, v) in logits.iter().enumerate() {
                if *v > logits[best] {
                    best = i;
                }
            }
            best as u32
        };
        if self.greedy() || logits.is_empty() {
            return argmax();
        }
        // Candidates, highest first: the top_k, or (top_k 0) those within 40 * max(1, T) of
        // the best logit (after the temperature, at most e^-40 of the best's weight each:
        // below f32's resolution of the sum).
        let c = &mut self.scratch;
        c.clear();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return argmax();
        }
        let floor = max - 40.0 * self.temperature.max(1.0);
        c.extend(
            logits
                .iter()
                .enumerate()
                .filter(|(_, v)| **v >= floor)
                .map(|(i, v)| (*v, i as u32)),
        );
        c.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        if self.top_k > 0 {
            c.truncate(self.top_k);
        }
        // Softmax at the temperature.
        let t = self.temperature;
        let top = c[0].0;
        let mut sum = 0f64;
        for x in c.iter_mut() {
            x.0 = ((x.0 - top) / t).exp();
            sum += x.0 as f64;
        }
        for x in c.iter_mut() {
            x.0 = (x.0 as f64 / sum) as f32;
        }
        // top_p: the smallest prefix reaching it.
        if self.top_p < 1.0 {
            let mut acc = 0f32;
            let mut keep = c.len();
            for (k, x) in c.iter().enumerate() {
                acc += x.0;
                if acc >= self.top_p {
                    keep = k + 1;
                    break;
                }
            }
            c.truncate(keep.max(1));
        }
        // min_p: at least min_p times the best's probability.
        if self.min_p > 0.0 {
            let bar = c[0].0 * self.min_p;
            let keep = c.iter().take_while(|x| x.0 >= bar).count();
            c.truncate(keep.max(1));
        }
        let total: f64 = c.iter().map(|x| x.0 as f64).sum();
        let c = std::mem::take(&mut self.scratch);
        let mut u = self.uniform() * total;
        let mut chosen = c[c.len() - 1].1;
        for x in c.iter() {
            u -= x.0 as f64;
            if u < 0.0 {
                chosen = x.1;
                break;
            }
        }
        self.scratch = c;
        chosen
    }
}

impl<'m> Engine<'m> {
    /// With `sampling` on, copies the pass's `rows` rows of logits (from `src`) for the host.
    pub(super) fn logits_out<'a>(&'a self, ops: &mut Vec<Op<'a>>, src: &'a Buffer, rows: usize) {
        if !self.sampling.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let words = rows * self.model.hp.vocab;
        ops.push(Op {
            kernel: &self.e.copyoff,
            buffers: vec![src, self.logits_host.buffer()],
            constants: vec![words as u32, 0, 0],
            groups: grid(words.div_ceil(256)),
            independent: false,
        });
        self.logits_rows
            .store(rows, std::sync::atomic::Ordering::Relaxed);
    }

    /// Row `row` of the last pass's logits (with `sampling`; `None`: the last row).
    pub(super) fn host_logits(&self, row: Option<usize>) -> Result<Vec<f32>> {
        let rows = self.logits_rows.load(std::sync::atomic::Ordering::Relaxed);
        let row = row.unwrap_or(rows.saturating_sub(1));
        if row >= rows {
            return Err("model_state_invalid");
        }
        let v = self.model.hp.vocab;
        let mut out = vec![0f32; v];
        self.gpu
            .read_mapped(&self.logits_host, (row * v * 4) as u64, &mut out)?;
        Ok(out)
    }

    /// The position (tokens in the sequence so far) and the most it may reach.
    pub fn position(&self) -> usize {
        self.pos
    }
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Saves the sequence as it is now (its recurrent states and position) for `rewind`.
    pub fn save_point(&mut self) -> Result<()> {
        let hp = &self.model.hp;
        let d = hp.kda_head * hp.heads;
        let state = hp.heads * 128 * 128;
        if self.saved.is_empty() {
            for blk in &self.blocks {
                self.saved.push(match blk.attention {
                    Attention::Kda { .. } => Some(self.gpu.buffer(((state + 9 * d) * 4) as u64)?),
                    _ => None,
                });
            }
            self.saved_last = Some(self.gpu.buffer((hp.embd * 4) as u64)?);
        }
        self.copy_point(true)?;
        self.saved_pos = Some(self.pos);
        Ok(())
    }

    /// Back to the saved point (false: none).
    pub fn rewind(&mut self) -> Result<bool> {
        let Some(pos) = self.saved_pos else {
            return Ok(false);
        };
        self.copy_point(false)?;
        self.pos = pos;
        Ok(true)
    }

    /// Forgets the saved point (a new sequence starts).
    pub fn forget_point(&mut self) {
        self.saved_pos = None;
    }

    fn copy_point(&self, save: bool) -> Result<()> {
        let hp = &self.model.hp;
        let d = hp.kda_head * hp.heads;
        let state = hp.heads * 128 * 128;
        let mut ops: Vec<Op> = Vec::new();
        for (blk, saved) in self.blocks.iter().zip(&self.saved) {
            let (
                Attention::Kda {
                    hist, state: st, ..
                },
                Some(saved),
            ) = (&blk.attention, saved)
            else {
                continue;
            };
            for (live, words, at) in [(st, state, 0), (hist, 9 * d, state)] {
                let (src, dst, so, d_o) = if save {
                    (live, saved, 0, at)
                } else {
                    (saved, live, at, 0)
                };
                ops.push(Op {
                    kernel: &self.e.copyoff,
                    buffers: vec![src, dst],
                    constants: vec![words as u32, so as u32, d_o as u32],
                    groups: grid(words.div_ceil(256)),
                    independent: true,
                });
            }
        }
        if let (Some(mtp), Some(saved)) = (&self.mtp, &self.saved_last) {
            let (src, dst) = if save {
                (&mtp.last, saved)
            } else {
                (saved, &mtp.last)
            };
            ops.push(Op {
                kernel: &self.e.copyoff,
                buffers: vec![src, dst],
                constants: vec![hp.embd as u32, 0, 0],
                groups: grid(hp.embd.div_ceil(256)),
                independent: true,
            });
        }
        if let Some(op) = ops.last_mut() {
            op.independent = false;
        }
        if ops.is_empty() {
            return Ok(());
        }
        submit(self, &mut ops)?.wait()
    }
}

#[cfg(test)]
mod tests {
    use super::Sampler;

    #[test]
    fn sampling_is_seeded_and_respects_its_limits() {
        let logits: Vec<f32> = (0..1000).map(|i| ((i * 37) % 101) as f32 / 10.0).collect();
        let draw = |seed| {
            let mut s = Sampler::new(seed, 1.0, 0, 0.95, 0.0);
            (0..64).map(|_| s.choose(&logits)).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
        // Greedy, and top_k 1, give the best (lowest index on ties).
        let best = Sampler::new(1, 0.0, 0, 1.0, 0.0).choose(&logits);
        assert_eq!(logits[best as usize], 10.0);
        let mut k1 = Sampler::new(3, 1.0, 1, 1.0, 0.0);
        assert!((0..32).all(|_| k1.choose(&logits) == best));
        // top_k 5: only the five best.
        let mut ranked: Vec<usize> = (0..logits.len()).collect();
        ranked.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
        let mut k5 = Sampler::new(4, 1.0, 5, 1.0, 0.0);
        assert!((0..256).all(|_| ranked[..5].contains(&(k5.choose(&logits) as usize))));
    }
}
