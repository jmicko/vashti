//! Qwen3-VL language models as text encoders: MiniMax H3's (Qwen3-VL 32B, of which the file
//! keeps the first 50 layers) and Qwen Image 2.1's (Qwen3-VL 8B, 36 layers). A prompt's
//! hidden states after the last layer, without the final norm, are the DiT's text features.
//! The sizes come from the file (`Config`).
//!
//! The vision tower stays with the C++ engine, which hands over each image's embeddings,
//! its patch grid and the deep-stack additions to the first layers. The linear layers are
//! INT8 in the rotated basis of the DiT's (`dit::linear8`).
//!
//! Qwen Image 2512's encoder is Qwen2.5-VL 7B (`TextEncoder::qwen25`): biases on queries,
//! keys and values, no query/key norms, plain rotary positions (theta 1e6; text tokens have
//! the same position on all three axes) and FP8 matrices with one scale each, which are
//! widened, rotated and rounded to INT8 per row here as the GPU form does.
//!
//! This is the CPU reference form: exact integer products, f32 elsewhere.
use std::path::Path;

use rayon::prelude::*;

use super::{
    dit::{linear8, quantize},
    safetensors::Safetensors,
};

/// The language model's sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub inner: usize,
}
impl Config {
    /// MiniMax H3's encoder (the first 50 layers of Qwen3-VL 32B).
    pub const MINIMAX_H3: Config = Config {
        width: 5120,
        layers: 50,
        heads: 64,
        kv_heads: 8,
        inner: 25600,
    };
    /// Qwen Image 2.1's encoder (Qwen3-VL 8B).
    pub const QWEN3_VL_8B: Config = Config {
        width: 4096,
        layers: 36,
        heads: 32,
        kv_heads: 8,
        inner: 12288,
    };
}
pub(super) const HEAD: usize = 128;
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
    /// `[size][width]`, replacing the placeholder tokens' embeddings.
    pub embeds: &'a [f32],
    /// One `[size][width]` addition after each of the first layers.
    pub deepstack: Vec<&'a [f32]>,
}
pub struct Prompt<'a> {
    pub ids: &'a [i32],
    pub images: Vec<Image<'a>>,
}
impl Prompt<'_> {
    pub fn check(&self, config: &Config) -> Result<(), &'static str> {
        let (width, layers) = (config.width, config.layers);
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
                || image.embeds.len() != image.size * width
                || image.deepstack.len() >= layers
                || image
                    .deepstack
                    .iter()
                    .any(|d| d.len() != image.size * width)
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

/// Qwen2.5-VL's rotary base.
const THETA_QWEN25: f32 = 1_000_000.0;
/// Cosine and sine of each token's channel pairs `[tokens][PAIRS]` for plain rotary
/// positions (one position per token).
pub fn rotary_plain(positions: &[[i32; 3]], theta: f32) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(positions.len() * PAIRS);
    for position in positions {
        for pair in 0..PAIRS {
            let frequency = theta.powf(-2.0 * pair as f32 / HEAD as f32);
            let angle = position[0] as f32 * frequency;
            out.push([angle.cos(), angle.sin()]);
        }
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
    pub config: Config,
    /// INT8 token embeddings in the rotated basis (with one scale per row), else BF16.
    int8_embeddings: bool,
    /// Rows of the embedding table.
    vocabulary: usize,
    /// Qwen2.5-VL (Qwen Image 2512's encoder) rather than Qwen3-VL.
    pub qwen25: bool,
    /// Accuracy experiments: float files' products in exact f32 instead of INT8.
    pub exact: bool,
}
impl TextEncoder {
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        let rows = |name: &str| -> Result<usize, &'static str> {
            let shape = file
                .shape(&format!("model.layers.0.{name}.weight"))
                .ok_or("model_file_invalid")?;
            shape.first().copied().ok_or("model_file_invalid")
        };
        let width = *file
            .shape("model.embed_tokens.weight")
            .and_then(|s| s.get(1))
            .ok_or("model_file_invalid")?;
        let config = Config {
            width,
            layers: file.count("model.layers."),
            heads: rows("self_attn.q_proj")? / HEAD,
            kv_heads: rows("self_attn.k_proj")? / HEAD,
            inner: rows("mlp.gate_proj")?,
        };
        let valid = config.layers > 0
            && config.kv_heads > 0
            && config.heads.is_multiple_of(config.kv_heads)
            && config.width.is_multiple_of(256)
            && config.inner.is_multiple_of(256);
        if !valid {
            return Err("model_file_invalid");
        }
        let int8_embeddings = file.shape("model.embed_tokens.weight_scale").is_some();
        let vocabulary = *file
            .shape("model.embed_tokens.weight")
            .and_then(|s| s.first())
            .ok_or("model_file_invalid")?;
        // Qwen2.5-VL has query/key/value biases (Qwen3-VL has query/key norms instead).
        let qwen25 = file.shape("model.layers.0.self_attn.q_proj.bias").is_some();
        if vocabulary < VOCABULARY {
            return Err("model_file_invalid");
        }
        Ok(Self {
            file,
            config,
            int8_embeddings,
            vocabulary,
            qwen25,
            exact: false,
        })
    }
    /// One token's embedding row.
    fn embedding(&self, id: usize) -> Result<Vec<f32>, &'static str> {
        let width = self.config.width;
        if !self.int8_embeddings {
            return self.file.bf16_rows(
                "model.embed_tokens.weight",
                [self.vocabulary, width],
                id,
                1,
            );
        }
        // Scaled bytes, then the inverse of the normalized Hadamard rotation, in the C++
        // engine's order (a 0.5 per radix-4 stage).
        let bytes = self
            .file
            .i8_row("model.embed_tokens.weight", [VOCABULARY, width], id)?;
        let scale = self
            .file
            .f32_row("model.embed_tokens.weight_scale", [VOCABULARY, 1], id)?[0];
        let mut row: Vec<f32> = bytes.iter().map(|&b| f32::from(b) * scale).collect();
        for group in row.chunks_exact_mut(256) {
            let mut stride = 1;
            while stride < 256 {
                for block in (0..256).step_by(4 * stride) {
                    for k in 0..stride {
                        let at = block + k;
                        let (a, b, c, d) = (
                            group[at],
                            group[at + stride],
                            group[at + 2 * stride],
                            group[at + 3 * stride],
                        );
                        group[at] = (a + b + c - d) * 0.5;
                        group[at + stride] = (a + b - c + d) * 0.5;
                        group[at + 2 * stride] = (a - b + c + d) * 0.5;
                        group[at + 3 * stride] = (-a + b + c + d) * 0.5;
                    }
                }
                stride *= 4;
            }
        }
        Ok(row)
    }
    /// Token embeddings with the images' rows spliced in, `[tokens][width]`.
    pub fn embed(&self, prompt: &Prompt) -> Result<Vec<f32>, &'static str> {
        let width = self.config.width;
        let mut x = Vec::with_capacity(prompt.ids.len() * width);
        for &id in prompt.ids {
            x.extend(self.embedding(id as usize)?);
        }
        for image in &prompt.images {
            x[image.index * width..(image.index + image.size) * width]
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
        if !self.is_int8(layer, name) {
            let weight = self.float_matrix(layer, name, rows, columns)?;
            return Ok(quantize(&weight, columns));
        }
        Ok((
            self.file.i8(&format!("{base}.weight"), &[rows, columns])?,
            self.file.f32(&format!("{base}.weight_scale"), &[rows, 1])?,
        ))
    }
    /// `x` times a layer's matrix: INT8 as the GPU computes it, or exact (`exact`).
    fn product(
        &self,
        layer: usize,
        x: &[f32],
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, &'static str> {
        if self.exact && !self.is_int8(layer, name) {
            let weight = self.float_matrix(layer, name, rows, columns)?;
            return Ok(super::dit::linear(x, columns, &weight, &[]));
        }
        let (weight, scale) = self.matrix(layer, name, rows, columns)?;
        Ok(linear8(x, columns, &weight, &scale))
    }
    /// Whether a layer's matrix is stored as INT8 (in the rotated basis, with row scales).
    pub(super) fn is_int8(&self, layer: usize, name: &str) -> bool {
        self.file
            .is_i8(&format!("model.layers.{layer}.{name}.weight"))
    }
    /// A float matrix widened (FP8 times its `scale_weight` when the file has one).
    pub(super) fn float_matrix(
        &self,
        layer: usize,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let base = format!("model.layers.{layer}.{name}");
        let mut weight = self
            .file
            .floats(&format!("{base}.weight"), &[rows, columns])?;
        if self.file.shape(&format!("{base}.scale_weight")).is_some() {
            let scale = self.file.f32(&format!("{base}.scale_weight"), &[])?[0];
            weight.par_iter_mut().for_each(|v| *v *= scale);
        }
        Ok(weight)
    }
    /// Qwen2.5-VL's outputs are the final norm's (Qwen3-VL's are taken before it).
    pub fn final_norm(&self, x: &mut [f32]) -> Result<(), &'static str> {
        if self.qwen25 {
            let width = self.config.width;
            let weight = self.file.floats("model.norm.weight", &[width])?;
            x.par_chunks_exact_mut(width)
                .for_each(|row| norm_in_place(row, &weight));
        }
        Ok(())
    }
    /// The rotary turns of a prompt's tokens.
    pub(super) fn turns(&self, prompt: &Prompt) -> Vec<[f32; 2]> {
        if self.qwen25 {
            rotary_plain(&positions(prompt), THETA_QWEN25)
        } else {
            rotary(&positions(prompt))
        }
    }
    /// Hidden states after the last layer, `[tokens][width]`.
    pub fn forward(&self, prompt: &Prompt) -> Result<Vec<f32>, &'static str> {
        prompt.check(&self.config)?;
        let width = self.config.width;
        let turns = self.turns(prompt);
        let mut x = self.embed(prompt)?;
        for layer in 0..self.config.layers {
            self.block(layer, &mut x, &turns)?;
            for image in &prompt.images {
                if let Some(extra) = image.deepstack.get(layer) {
                    let rows = &mut x[image.index * width..(image.index + image.size) * width];
                    rows.iter_mut().zip(*extra).for_each(|(v, e)| *v += e);
                }
            }
        }
        self.final_norm(&mut x)?;
        Ok(x)
    }
    fn block(&self, layer: usize, x: &mut [f32], turns: &[[f32; 2]]) -> Result<(), &'static str> {
        let Config {
            width,
            heads,
            kv_heads,
            inner,
            ..
        } = self.config;
        let n = x.len() / width;
        let h = rms_norm(x, &self.vector(layer, "input_layernorm", width)?);
        let project = |name: &str, rows: usize| -> Result<Vec<f32>, &'static str> {
            self.product(layer, &h, name, rows, width)
        };
        let mut q = project("self_attn.q_proj", heads * HEAD)?;
        let mut k = project("self_attn.k_proj", kv_heads * HEAD)?;
        let mut v = project("self_attn.v_proj", kv_heads * HEAD)?;
        if self.qwen25 {
            for (values, name) in [
                (&mut q, "self_attn.q_proj"),
                (&mut k, "self_attn.k_proj"),
                (&mut v, "self_attn.v_proj"),
            ] {
                let rows = values.len() / n;
                let bias = self
                    .file
                    .floats(&format!("model.layers.{layer}.{name}.bias"), &[rows])?;
                values.par_chunks_exact_mut(rows).for_each(|row| {
                    row.iter_mut().zip(&bias).for_each(|(v, b)| *v += b);
                });
            }
        }
        for (values, name) in [(&mut q, "self_attn.q_norm"), (&mut k, "self_attn.k_norm")] {
            let weight = if self.qwen25 {
                None
            } else {
                Some(self.vector(layer, name, HEAD)?)
            };
            let heads = values.len() / n / HEAD;
            values
                .par_chunks_exact_mut(HEAD)
                .enumerate()
                .for_each(|(i, head)| {
                    if let Some(weight) = &weight {
                        norm_in_place(head, weight);
                    }
                    rotate(head, &turns[i / heads * PAIRS..][..PAIRS]);
                });
        }
        let mixed = attend(&q, &k, &v, n, heads, kv_heads);
        let out = self.product(layer, &mixed, "self_attn.o_proj", width, heads * HEAD)?;
        x.iter_mut().zip(&out).for_each(|(x, o)| *x += o);

        let h = rms_norm(x, &self.vector(layer, "post_attention_layernorm", width)?);
        let mut gate = self.product(layer, &h, "mlp.gate_proj", inner, width)?;
        let up = self.product(layer, &h, "mlp.up_proj", inner, width)?;
        gate.iter_mut()
            .zip(&up)
            .for_each(|(g, u)| *g = *g / (1.0 + (-*g).exp()) * u);
        let out = self.product(layer, &gate, "mlp.down_proj", width, inner)?;
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
/// Causal attention with grouped keys and values; returns `[tokens][heads * HEAD]`.
fn attend(q: &[f32], k: &[f32], v: &[f32], n: usize, heads: usize, kv_heads: usize) -> Vec<f32> {
    let scale = 1.0 / (HEAD as f32).sqrt();
    let mut out = vec![0f32; n * heads * HEAD];
    out.par_chunks_exact_mut(HEAD)
        .enumerate()
        .for_each(|(i, out)| {
            let (token, head) = (i / heads, i % heads);
            let group = head / (heads / kv_heads);
            let query = &q[(token * heads + head) * HEAD..][..HEAD];
            let scores: Vec<f32> = (0..=token)
                .map(|t| {
                    let key = &k[(t * kv_heads + group) * HEAD..][..HEAD];
                    query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let top = scores.iter().fold(f32::NEG_INFINITY, |m, s| m.max(*s));
            let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let total: f32 = weights.iter().sum();
            for (t, w) in weights.iter().enumerate() {
                let value = &v[(t * kv_heads + group) * HEAD..][..HEAD];
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
                    size: self.grid.0 * self.grid.1 / (MERGE * MERGE),
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
        let embeds = vec![0f32; 6 * Config::MINIMAX_H3.width];
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
        prompt.check(&Config::MINIMAX_H3).unwrap();
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
