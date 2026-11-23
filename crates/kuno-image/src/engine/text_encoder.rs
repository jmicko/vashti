//! MiniMax H3 text encoder: the language model of Qwen3-VL 32B, of which the file keeps the
//! first 50 layers. A prompt's hidden states after the last layer, without the final norm,
//! are the DiT's text features.
//!
//! The vision tower stays with the C++ engine, which hands over each image's embeddings,
//! its patch grid and the deep-stack additions to the first layers. The linear layers are
//! INT8 in the rotated basis of the DiT's (`dit::linear8`).
//!
//! This is the CPU reference form: exact integer products, f32 elsewhere.
use std::path::Path;

use rayon::prelude::*;

use super::{dit::linear8, safetensors::Safetensors};

pub const WIDTH: usize = 5120;
pub const LAYERS: usize = 50;
pub(super) const HEADS: usize = 64;
pub(super) const KV_HEADS: usize = 8;
pub(super) const HEAD: usize = 128;
pub(super) const INNER: usize = 25600;
pub(super) const VOCABULARY: usize = 151936;
const EPS: f32 = 1e-6;
const THETA: f32 = 5_000_000.0;
/// Channel pairs per head and the interleaved axis sections (time, height, width).
pub(super) const PAIRS: usize = HEAD / 2;
const SECTIONS: [usize; 3] = [24, 20, 20];
/// Vision tokens merge 2 x 2 patches.
const MERGE: usize = 2;

/// One image of a prompt: its first token, patch grid and the vision tower's outputs.
pub struct Image<'a> {
    pub index: usize,
    pub size: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    /// `[size][WIDTH]`, replacing the placeholder tokens' embeddings.
    pub embeds: &'a [f32],
    /// One `[size][WIDTH]` addition after each of the first layers.
    pub deepstack: Vec<&'a [f32]>,
}
pub struct Prompt<'a> {
    pub ids: &'a [i32],
    pub images: Vec<Image<'a>>,
}
impl Prompt<'_> {
    pub fn check(&self) -> Result<(), &'static str> {
        let invalid = Err("text_invalid");
        if self.ids.is_empty()
            || self
                .ids
                .iter()
                .any(|id| !(0..VOCABULARY as i32).contains(id))
        {
            return invalid;
        }
        let mut end = 0;
        for image in &self.images {
            let (gh, gw) = (image.grid_h / MERGE, image.grid_w / MERGE);
            if image.index < end
                || image.index + image.size > self.ids.len()
                || image.size == 0
                || gh * gw != image.size
                || image.embeds.len() != image.size * WIDTH
                || image.deepstack.len() >= LAYERS
                || image
                    .deepstack
                    .iter()
                    .any(|d| d.len() != image.size * WIDTH)
            {
                return invalid;
            }
            end = image.index + image.size;
        }
        Ok(())
    }
}

/// Rotary positions (time, height, width) of every token, as the C++ engine assigns them.
pub fn positions(prompt: &Prompt) -> Vec<[i32; 3]> {
    let n = prompt.ids.len();
    let mut out: Vec<[i32; 3]> = (0..n as i32).map(|i| [i, i, i]).collect();
    let mut offset = 0i32;
    for image in &prompt.images {
        let (index, size) = (image.index as i32, image.size as i32);
        let end = image.index + image.size;
        let (gh, gw) = ((image.grid_h / MERGE) as i32, (image.grid_w / MERGE) as i32);
        let longest = gh.max(gw);
        let next = index + longest + offset;
        for (token, at) in out.iter_mut().enumerate().skip(end) {
            let p = next + token as i32 - end as i32;
            *at = [p, p, p];
        }
        for token in 0..size {
            out[image.index + token as usize] = [
                index + offset,
                index + offset + token / gw,
                index + offset + token % gw,
            ];
        }
        offset += longest - size;
    }
    out
}

/// Cosine and sine of each token's channel pairs, `[tokens][PAIRS]`. Pairs are interleaved
/// over the axes (time, height, width, time, ...); the last two use a fourth position that
/// is always 0, as in ggml's IMROPE.
pub fn rotary(positions: &[[i32; 3]]) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(positions.len() * PAIRS);
    let scale = THETA.powf(-2.0 / HEAD as f32);
    for position in positions {
        let mut frequency = 1.0f32;
        for pair in 0..PAIRS {
            let p = match pair % 3 {
                0 if pair < 3 * SECTIONS[0] => position[0],
                1 if pair < 3 * SECTIONS[1] => position[1],
                2 if pair < 3 * SECTIONS[2] => position[2],
                _ => 0,
            };
            let angle = p as f32 * frequency;
            out.push([angle.cos(), angle.sin()]);
            frequency *= scale;
        }
    }
    out
}

pub struct TextEncoder {
    pub(super) file: Safetensors,
}
impl TextEncoder {
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        if file.count("model.layers.") != LAYERS {
            return Err("model_file_invalid");
        }
        Ok(Self { file })
    }
    /// Token embeddings with the images' rows spliced in, `[tokens][WIDTH]`.
    pub fn embed(&self, prompt: &Prompt) -> Result<Vec<f32>, &'static str> {
        let mut x = Vec::with_capacity(prompt.ids.len() * WIDTH);
        for &id in prompt.ids {
            x.extend(self.file.bf16_rows(
                "model.embed_tokens.weight",
                [VOCABULARY, WIDTH],
                id as usize,
                1,
            )?);
        }
        for image in &prompt.images {
            x[image.index * WIDTH..(image.index + image.size) * WIDTH]
                .copy_from_slice(image.embeds);
        }
        Ok(x)
    }
    fn vector(&self, layer: usize, name: &str, width: usize) -> Result<Vec<f32>, &'static str> {
        self.file
            .floats(&format!("model.layers.{layer}.{name}.weight"), &[width])
    }
    fn matrix(
        &self,
        layer: usize,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<(Vec<i8>, Vec<f32>), &'static str> {
        let base = format!("model.layers.{layer}.{name}");
        Ok((
            self.file.i8(&format!("{base}.weight"), &[rows, columns])?,
            self.file.f32(&format!("{base}.weight_scale"), &[rows, 1])?,
        ))
    }
    /// Hidden states after the last layer, `[tokens][WIDTH]`.
    pub fn forward(&self, prompt: &Prompt) -> Result<Vec<f32>, &'static str> {
        prompt.check()?;
        let turns = rotary(&positions(prompt));
        let mut x = self.embed(prompt)?;
        for layer in 0..LAYERS {
            self.block(layer, &mut x, &turns)?;
            for image in &prompt.images {
                if let Some(extra) = image.deepstack.get(layer) {
                    let rows = &mut x[image.index * WIDTH..(image.index + image.size) * WIDTH];
                    rows.iter_mut().zip(*extra).for_each(|(v, e)| *v += e);
                }
            }
        }
        Ok(x)
    }
    fn block(&self, layer: usize, x: &mut [f32], turns: &[[f32; 2]]) -> Result<(), &'static str> {
        let n = x.len() / WIDTH;
        let h = rms_norm(x, &self.vector(layer, "input_layernorm", WIDTH)?);
        let project = |name: &str, rows: usize| -> Result<Vec<f32>, &'static str> {
            let (weight, scale) = self.matrix(layer, name, rows, WIDTH)?;
            Ok(linear8(&h, WIDTH, &weight, &scale))
        };
        let mut q = project("self_attn.q_proj", HEADS * HEAD)?;
        let mut k = project("self_attn.k_proj", KV_HEADS * HEAD)?;
        let v = project("self_attn.v_proj", KV_HEADS * HEAD)?;
        for (values, name) in [(&mut q, "self_attn.q_norm"), (&mut k, "self_attn.k_norm")] {
            let weight = self.vector(layer, name, HEAD)?;
            let heads = values.len() / n / HEAD;
            values
                .par_chunks_exact_mut(HEAD)
                .enumerate()
                .for_each(|(i, head)| {
                    norm_in_place(head, &weight);
                    rotate(head, &turns[i / heads * PAIRS..][..PAIRS]);
                });
        }
        let mixed = attend(&q, &k, &v, n);
        let (weight, scale) = self.matrix(layer, "self_attn.o_proj", WIDTH, HEADS * HEAD)?;
        let out = linear8(&mixed, HEADS * HEAD, &weight, &scale);
        x.iter_mut().zip(&out).for_each(|(x, o)| *x += o);

        let h = rms_norm(x, &self.vector(layer, "post_attention_layernorm", WIDTH)?);
        let (weight, scale) = self.matrix(layer, "mlp.gate_proj", INNER, WIDTH)?;
        let mut gate = linear8(&h, WIDTH, &weight, &scale);
        let (weight, scale) = self.matrix(layer, "mlp.up_proj", INNER, WIDTH)?;
        let up = linear8(&h, WIDTH, &weight, &scale);
        gate.iter_mut()
            .zip(&up)
            .for_each(|(g, u)| *g = *g / (1.0 + (-*g).exp()) * u);
        let (weight, scale) = self.matrix(layer, "mlp.down_proj", WIDTH, INNER)?;
        let out = linear8(&gate, INNER, &weight, &scale);
        x.iter_mut().zip(&out).for_each(|(x, o)| *x += o);
        Ok(())
    }
}

fn norm_in_place(row: &mut [f32], weight: &[f32]) {
    let mean = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let scale = 1.0 / (mean + EPS).sqrt();
    row.iter_mut()
        .zip(weight)
        .for_each(|(v, w)| *v = *v * scale * w);
}
fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    out.par_chunks_exact_mut(weight.len())
        .for_each(|row| norm_in_place(row, weight));
    out
}
/// Rotates channel `j` with channel `j + PAIRS` (NEOX pairing).
fn rotate(head: &mut [f32], turns: &[[f32; 2]]) {
    let (low, high) = head.split_at_mut(PAIRS);
    for ((a, b), [cos, sin]) in low.iter_mut().zip(high).zip(turns) {
        let (x0, x1) = (*a, *b);
        *a = x0 * cos - x1 * sin;
        *b = x0 * sin + x1 * cos;
    }
}
/// Causal attention with grouped keys and values; returns `[tokens][HEADS * HEAD]`.
fn attend(q: &[f32], k: &[f32], v: &[f32], n: usize) -> Vec<f32> {
    let scale = 1.0 / (HEAD as f32).sqrt();
    let mut out = vec![0f32; n * HEADS * HEAD];
    out.par_chunks_exact_mut(HEAD)
        .enumerate()
        .for_each(|(i, out)| {
            let (token, head) = (i / HEADS, i % HEADS);
            let group = head / (HEADS / KV_HEADS);
            let query = &q[(token * HEADS + head) * HEAD..][..HEAD];
            let scores: Vec<f32> = (0..=token)
                .map(|t| {
                    let key = &k[(t * KV_HEADS + group) * HEAD..][..HEAD];
                    query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let top = scores.iter().fold(f32::NEG_INFINITY, |m, s| m.max(*s));
            let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let total: f32 = weights.iter().sum();
            for (t, w) in weights.iter().enumerate() {
                let value = &v[(t * KV_HEADS + group) * HEAD..][..HEAD];
                out.iter_mut().zip(value).for_each(|(o, x)| *o += w * x);
            }
            out.iter_mut().for_each(|o| *o /= total);
        });
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A made-up prompt: deterministic token ids and one image whose embeddings and
    /// deep-stack additions are other tokens' embedding rows, so magnitudes are realistic.
    pub(crate) struct Synthetic {
        pub ids: Vec<i32>,
        pub index: usize,
        pub grid: (usize, usize),
        pub embeds: Vec<f32>,
        pub deepstack: Vec<Vec<f32>>,
    }
    impl Synthetic {
        pub fn new(encoder: &TextEncoder, tokens: usize, grid: (usize, usize)) -> Self {
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 33) as usize % 151_000
            };
            let ids: Vec<i32> = (0..tokens).map(|_| next() as i32).collect();
            let size = grid.0 * grid.1 / (MERGE * MERGE);
            let rows = |count: usize, next: &mut dyn FnMut() -> usize| {
                let ids: Vec<i32> = (0..count).map(|_| next() as i32).collect();
                encoder
                    .embed(&Prompt {
                        ids: &ids,
                        images: vec![],
                    })
                    .unwrap()
            };
            let embeds = rows(size, &mut next);
            let deepstack = (0..3)
                .map(|_| rows(size, &mut next).iter().map(|v| v * 0.5).collect())
                .collect();
            Self {
                ids,
                index: 5,
                grid,
                embeds,
                deepstack,
            }
        }
        pub fn prompt(&self) -> Prompt<'_> {
            Prompt {
                ids: &self.ids,
                images: vec![Image {
                    index: self.index,
                    size: self.embeds.len() / WIDTH,
                    grid_h: self.grid.0,
                    grid_w: self.grid.1,
                    embeds: &self.embeds,
                    deepstack: self.deepstack.iter().map(Vec::as_slice).collect(),
                }],
            }
        }
    }
    pub(crate) fn snr(got: &[f32], want: &[f32]) -> f64 {
        let (mut error, mut power) = (0f64, 0f64);
        for (g, w) in got.iter().zip(want) {
            error += f64::from(g - w).powi(2);
            power += f64::from(*w).powi(2);
        }
        10.0 * (power / error).log10()
    }

    #[test]
    fn text_positions_follow_the_image_grid() {
        let ids = vec![1i32; 10];
        let embeds = vec![0f32; 6 * WIDTH];
        let prompt = Prompt {
            ids: &ids,
            images: vec![Image {
                index: 2,
                size: 6,
                grid_h: 4,
                grid_w: 6,
                embeds: &embeds,
                deepstack: vec![],
            }],
        };
        prompt.check().unwrap();
        let p = positions(&prompt);
        assert_eq!(p[1], [1, 1, 1]);
        assert_eq!(p[2], [2, 2, 2]);
        assert_eq!(p[4], [2, 2, 4]);
        assert_eq!(p[5], [2, 3, 2]);
        // After the image: index + longest side.
        assert_eq!(p[8], [5, 5, 5]);
        assert_eq!(p[9], [6, 6, 6]);
        let turns = rotary(&p);
        assert_eq!(turns[PAIRS + 61], [1.0, 0.0]);
    }
}
