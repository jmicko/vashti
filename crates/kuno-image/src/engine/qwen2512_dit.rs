//! Qwen Image 2512's diffusion transformer on the CPU: the reference the GPU form is checked
//! against, written from the C++ engine's graph (`qwen_image.hpp`).
//!
//! Two streams of 3072-wide tokens, the prompt's (text) and the target's (image), each with
//! its own weights in all 60 blocks. A block modulates each stream's LayerNorm (no weights)
//! with `x * (1 + scale) + shift` from its own six vectors of the time embedding, runs one
//! joint attention over text then image tokens (RMS-normed queries and keys, three-axis
//! rotary positions, no mask), and adds each stream's projected attention and GELU MLP back
//! gated. Text tokens see the image, so nothing can be computed once per prompt. The file's
//! tensors are all FP8 (e4m3, no scales); here they are widened exactly to f32.

use std::path::Path;

use rayon::prelude::*;

use super::{
    dit::{linear, linear8, quantize},
    qwen_dit::{gelu, rotary, time_embedding},
    safetensors::Safetensors,
};

/// Values per token: 16 latent channels times a 2 x 2 patch.
pub const CHANNELS: usize = 64;
pub const LATENT: usize = 16;
/// Width of the text encoder's rows.
pub const CONTEXT: usize = 3584;
pub(super) const DIM: usize = 3072;
pub(super) const HEAD: usize = 128;
pub(super) const HEADS: usize = DIM / HEAD;
pub(super) const HIDDEN: usize = 12288;
pub(super) const PAIRS: usize = HEAD / 2;
const TIME: usize = 256;
const EPS: f32 = 1e-6;
const PREFIX: &str = "model.diffusion_model.";
/// The prompt template of text-to-image and how many of its leading tokens are dropped.
pub const TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
pub const START: usize = 34;

/// Rotary positions: text tokens first (`s, s, s` from `max(h, w) / 2`), then the image's
/// rows centered on zero (`0, y - h / 2, x - w / 2`), `h` and `w` in tokens.
pub fn positions(text: usize, height: usize, width: usize) -> Vec<[f32; 3]> {
    let start = (height.max(width) / 2) as f32;
    let mut out: Vec<[f32; 3]> = (0..text).map(|j| [start + j as f32; 3]).collect();
    let (h0, w0) = ((height / 2) as f32, (width / 2) as f32);
    for y in 0..height {
        for x in 0..width {
            out.push([0.0, y as f32 - h0, x as f32 - w0]);
        }
    }
    out
}

/// `[LATENT][2h][2w]` to `[h * w][CHANNELS]`, a token's values in (channel, row, column)
/// order of its 2 x 2 patch.
pub fn patchify(latent: &[f32], height: usize, width: usize) -> Vec<f32> {
    let (rows, columns) = (2 * height, 2 * width);
    let mut out = vec![0f32; height * width * CHANNELS];
    for (cell, token) in out.chunks_exact_mut(CHANNELS).enumerate() {
        let (y, x) = (cell / width, cell % width);
        for (i, v) in token.iter_mut().enumerate() {
            let (c, py, px) = (i / 4, i / 2 % 2, i % 2);
            *v = latent[(c * rows + 2 * y + py) * columns + 2 * x + px];
        }
    }
    out
}
pub fn unpatchify(tokens: &[f32], height: usize, width: usize) -> Vec<f32> {
    let (rows, columns) = (2 * height, 2 * width);
    let mut out = vec![0f32; LATENT * rows * columns];
    for (cell, token) in tokens.chunks_exact(CHANNELS).enumerate() {
        let (y, x) = (cell / width, cell % width);
        for (i, v) in token.iter().enumerate() {
            let (c, py, px) = (i / 4, i / 2 % 2, i % 2);
            out[(c * rows + 2 * y + py) * columns + 2 * x + px] = *v;
        }
    }
    out
}

pub(super) fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}
/// Rows of `DIM` to zero mean and unit variance (no weights).
fn layer_norm(x: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    out.par_chunks_exact_mut(DIM).for_each(|row| {
        let mean = row.iter().sum::<f32>() / DIM as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / DIM as f32;
        let scale = 1.0 / (var + EPS).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) * scale;
        }
    });
    out
}
fn rms_norm(row: &mut [f32], weight: &[f32]) {
    let mean = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let scale = 1.0 / (mean + EPS).sqrt();
    for (v, w) in row.iter_mut().zip(weight) {
        *v *= scale * w;
    }
}

/// Inputs of one evaluation.
pub struct Step<'a> {
    /// `[LATENT][2 * height][2 * width]`.
    pub x: &'a [f32],
    /// In tokens (latent size over two).
    pub height: usize,
    pub width: usize,
    /// On the `[0, 1000]` scale.
    pub timestep: f32,
    /// Text encoder rows `[tokens][CONTEXT]`.
    pub context: &'a [f32],
}

pub struct Qwen2512Dit {
    pub(super) file: Safetensors,
    pub(super) layers: usize,
    /// Block linears as the GPU runs them: weights rotated in groups of 256 and rounded to
    /// int8 per output row, activations the same per token (accuracy experiments).
    pub(super) int8: bool,
    /// Accuracy experiments for fast attention, bits: queries (1) and keys (2) rounded to
    /// int8 per token, each head's mean key subtracted first (4), not in blocks with outlier
    /// norm weights (8).
    pub(super) attention8: u8,
}
impl Qwen2512Dit {
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        let layers = file.count(&format!("{PREFIX}transformer_blocks."));
        if layers == 0 {
            return Err("model_file_invalid");
        }
        Ok(Self {
            file,
            layers,
            int8: false,
            attention8: 0,
        })
    }
    pub(super) fn floats(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, &'static str> {
        self.file.floats(&format!("{PREFIX}{name}"), shape)
    }
    /// `x W^T + b` with the layer's weight `[outputs][inputs]` and bias.
    fn dense(
        &self,
        x: &[f32],
        name: &str,
        outputs: usize,
        inputs: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let weight = self.floats(&format!("{name}.weight"), &[outputs, inputs])?;
        let bias = self.floats(&format!("{name}.bias"), &[outputs])?;
        Ok(linear(x, inputs, &weight, &bias))
    }
    /// A block linear: as `dense`, or through int8 (`int8`).
    fn block_dense(
        &self,
        x: &[f32],
        name: &str,
        outputs: usize,
        inputs: usize,
    ) -> Result<Vec<f32>, &'static str> {
        if !self.int8 {
            return self.dense(x, name, outputs, inputs);
        }
        let weight = self.floats(&format!("{name}.weight"), &[outputs, inputs])?;
        let bias = self.floats(&format!("{name}.bias"), &[outputs])?;
        let (bytes, scales) = quantize(&weight, inputs);
        let mut out = linear8(x, inputs, &bytes, &scales);
        out.par_chunks_exact_mut(outputs).for_each(|row| {
            for (v, b) in row.iter_mut().zip(&bias) {
                *v += b;
            }
        });
        Ok(out)
    }

    /// `silu(time embedding)`, the input of every modulation layer: `[DIM]`.
    pub(super) fn time(&self, timestep: f32) -> Result<Vec<f32>, &'static str> {
        let name = "time_text_embed.timestep_embedder";
        let mut h = self.dense(
            &time_embedding(timestep),
            &format!("{name}.linear_1"),
            DIM,
            TIME,
        )?;
        h.iter_mut().for_each(|v| *v = silu(*v));
        let mut h = self.dense(&h, &format!("{name}.linear_2"), DIM, DIM)?;
        h.iter_mut().for_each(|v| *v = silu(*v));
        Ok(h)
    }
    /// Text encoder rows to the stream's width.
    pub(super) fn text(&self, context: &[f32]) -> Result<Vec<f32>, &'static str> {
        let weight = self.floats("txt_norm.weight", &[CONTEXT])?;
        let mut x = context.to_vec();
        x.par_chunks_exact_mut(CONTEXT)
            .for_each(|row| rms_norm(row, &weight));
        self.dense(&x, "txt_in", DIM, CONTEXT)
    }

    pub fn forward(&self, step: &Step) -> Result<Vec<f32>, &'static str> {
        let text_len = step.context.len() / CONTEXT;
        let time = self.time(step.timestep)?;
        let mut txt = self.text(step.context)?;
        let mut img = self.dense(
            &patchify(step.x, step.height, step.width),
            "img_in",
            DIM,
            CHANNELS,
        )?;
        let rotary = rotary(&positions(text_len, step.height, step.width));
        for layer in 0..self.layers {
            (img, txt) = self.block(layer, img, txt, &time, &rotary)?;
        }
        // AdaLayerNormContinuous: scale first, then shift.
        let mods = self.dense(&time, "norm_out.linear", 2 * DIM, DIM)?;
        let (scale, shift) = mods.split_at(DIM);
        let mut h = layer_norm(&img);
        h.par_chunks_exact_mut(DIM).for_each(|row| {
            for ((v, s), b) in row.iter_mut().zip(scale).zip(shift) {
                *v = *v * (1.0 + s) + b;
            }
        });
        let out = self.dense(&h, "proj_out", CHANNELS, DIM)?;
        Ok(unpatchify(&out, step.height, step.width))
    }

    /// One stream's six modulation vectors: shift, scale, gate (attention), then the MLP's.
    fn modulation(&self, name: &str, time: &[f32]) -> Result<Vec<f32>, &'static str> {
        self.dense(time, name, 6 * DIM, DIM)
    }

    fn block(
        &self,
        layer: usize,
        mut img: Vec<f32>,
        mut txt: Vec<f32>,
        time: &[f32],
        rotary: &[[f32; 2]],
    ) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
        let name = format!("transformer_blocks.{layer}");
        let img_mod = self.modulation(&format!("{name}.img_mod.1"), time)?;
        let txt_mod = self.modulation(&format!("{name}.txt_mod.1"), time)?;
        let part = |m: &[f32], i: usize| m[i * DIM..(i + 1) * DIM].to_vec();
        let modulate = |x: &[f32], m: &[f32], i: usize| {
            let (shift, scale) = (part(m, i), part(m, i + 1));
            let mut h = layer_norm(x);
            h.par_chunks_exact_mut(DIM).for_each(|row| {
                for ((v, s), b) in row.iter_mut().zip(&scale).zip(&shift) {
                    *v = *v * (1.0 + s) + b;
                }
            });
            h
        };
        let gated_add = |x: &mut [f32], y: &[f32], gate: &[f32]| {
            x.par_chunks_exact_mut(DIM)
                .zip(y.par_chunks_exact(DIM))
                .for_each(|(x, y)| {
                    for ((x, y), g) in x.iter_mut().zip(y).zip(gate) {
                        *x += y * g;
                    }
                });
        };

        let (hi, ht) = (modulate(&img, &img_mod, 0), modulate(&txt, &txt_mod, 0));
        let a = format!("{name}.attn");
        let project =
            |h: &[f32], which: &str| self.block_dense(h, &format!("{a}.{which}"), DIM, DIM);
        // Joint sequence: text then image.
        let mut q = project(&ht, "add_q_proj")?;
        q.extend(project(&hi, "to_q")?);
        let mut k = project(&ht, "add_k_proj")?;
        k.extend(project(&hi, "to_k")?);
        let mut v = project(&ht, "add_v_proj")?;
        v.extend(project(&hi, "to_v")?);
        let norm = |which: &str| self.floats(&format!("{a}.{which}.weight"), &[HEAD]);
        let text_len = txt.len() / DIM;
        let norms = [
            norm("norm_added_q")?,
            norm("norm_q")?,
            norm("norm_added_k")?,
            norm("norm_k")?,
        ];
        let mode = if self.attention8 & 8 != 0 && outlier_norms(&norms) {
            0
        } else {
            self.attention8 & 7
        };
        let attended = attention(
            &q,
            &k,
            &v,
            text_len,
            [&norms[0], &norms[1]],
            [&norms[2], &norms[3]],
            rotary,
            mode,
        );
        let (at, ai) = attended.split_at(text_len * DIM);
        let out_i = self.block_dense(ai, &format!("{a}.to_out.0"), DIM, DIM)?;
        let out_t = self.block_dense(at, &format!("{a}.to_add_out"), DIM, DIM)?;
        gated_add(&mut img, &out_i, &part(&img_mod, 2));
        gated_add(&mut txt, &out_t, &part(&txt_mod, 2));

        let mlp = |x: &[f32], m: &[f32], stream: &str| -> Result<Vec<f32>, &'static str> {
            let h = modulate(x, m, 3);
            let mut h =
                self.block_dense(&h, &format!("{name}.{stream}.net.0.proj"), HIDDEN, DIM)?;
            h.par_iter_mut().for_each(|v| *v = gelu(*v));
            self.block_dense(&h, &format!("{name}.{stream}.net.2"), DIM, HIDDEN)
        };
        let out_i = mlp(&img, &img_mod, "img_mlp")?;
        let out_t = mlp(&txt, &txt_mod, "txt_mlp")?;
        gated_add(&mut img, &out_i, &part(&img_mod, 5));
        gated_add(&mut txt, &out_t, &part(&txt_mod, 5));
        Ok((img, txt))
    }
}

/// Whether a block's query/key norm weights have an outlier channel (largest magnitude over
/// three times the median): one channel then dominates the scores, and int8 queries and keys
/// rounded per token lose the rest. Such blocks keep half-float attention in fast mode.
pub fn outlier_norms(norms: &[Vec<f32>]) -> bool {
    norms.iter().any(|w| {
        let mut sorted: Vec<f32> = w.iter().map(|v| v.abs()).collect();
        sorted.sort_by(f32::total_cmp);
        sorted[sorted.len() - 1] > 3.0 * sorted[sorted.len() / 2]
    })
}

/// Multi-head attention over the joint sequence (no mask). `q`, `k`, `v` are
/// `[tokens][DIM]`, the first `text` rows text; queries and keys get their stream's
/// weighted RMS norm (`[text, image]`) and the rotary positions.
#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    text: usize,
    qw: [&[f32]; 2],
    kw: [&[f32]; 2],
    rotary: &[[f32; 2]],
    mode: u8,
) -> Vec<f32> {
    let round = |x: &mut [f32]| {
        for row in x.chunks_exact_mut(HEAD) {
            let top = row.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
            let scale = top / 127.0;
            for v in row.iter_mut() {
                *v = (*v / scale).round_ties_even() * scale;
            }
        }
    };
    let tokens = q.len() / DIM;
    let prepare = |x: &[f32], weight: [&[f32]; 2], head: usize| {
        let mut out = vec![0f32; tokens * HEAD];
        for (token, row) in out.chunks_exact_mut(HEAD).enumerate() {
            row.copy_from_slice(&x[token * DIM + head * HEAD..][..HEAD]);
            rms_norm(row, weight[usize::from(token >= text)]);
            for (pair, [cos, sin]) in rotary[token * PAIRS..][..PAIRS].iter().enumerate() {
                let (a, b) = (row[2 * pair], row[2 * pair + 1]);
                row[2 * pair] = cos * a - sin * b;
                row[2 * pair + 1] = sin * a + cos * b;
            }
        }
        out
    };
    let scale = 1.0 / (HEAD as f32).sqrt();
    let heads: Vec<Vec<f32>> = (0..HEADS)
        .into_par_iter()
        .map(|head| {
            let (mut query, mut key) = (prepare(q, qw, head), prepare(k, kw, head));
            if mode & 4 != 0 {
                let mut mean = vec![0f32; HEAD];
                for row in key.chunks_exact(HEAD) {
                    for (m, v) in mean.iter_mut().zip(row) {
                        *m += v / tokens as f32;
                    }
                }
                for row in key.chunks_exact_mut(HEAD) {
                    for (v, m) in row.iter_mut().zip(&mean) {
                        *v -= m;
                    }
                }
            }
            if mode & 1 != 0 {
                round(&mut query);
            }
            if mode & 2 != 0 {
                round(&mut key);
            }
            let mut out = vec![0f32; tokens * HEAD];
            let mut scores = vec![0f32; tokens];
            for t in 0..tokens {
                let qrow = &query[t * HEAD..][..HEAD];
                let mut top = f32::NEG_INFINITY;
                for (j, s) in scores.iter_mut().enumerate() {
                    let krow = &key[j * HEAD..][..HEAD];
                    *s = qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale;
                    top = top.max(*s);
                }
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - top).exp();
                    sum += *s;
                }
                let row = &mut out[t * HEAD..][..HEAD];
                for (j, s) in scores.iter().enumerate() {
                    let w = s / sum;
                    let vrow = &v[j * DIM + head * HEAD..][..HEAD];
                    for (o, v) in row.iter_mut().zip(vrow) {
                        *o += w * v;
                    }
                }
            }
            out
        })
        .collect();
    let mut out = vec![0f32; tokens * DIM];
    for (head, values) in heads.iter().enumerate() {
        for (token, row) in values.chunks_exact(HEAD).enumerate() {
            out[token * DIM + head * HEAD..][..HEAD].copy_from_slice(row);
        }
    }
    out
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::engine::qwen_dit::tests::{db, read};

    /// One captured C++ call: `<dir>/step<N>.{x,timestep,context,out}.{f32,txt}`, shapes in
    /// ggml order (innermost first).
    pub(crate) struct Golden {
        pub x: Vec<f32>,
        /// In tokens.
        pub height: usize,
        pub width: usize,
        pub timestep: f32,
        pub context: Vec<f32>,
        pub out: Vec<f32>,
    }
    impl Golden {
        pub(crate) fn load(dir: &Path, step: usize) -> Self {
            let file = |name: &str, ext: &str| dir.join(format!("step{step}.{name}.{ext}"));
            let shape = |name: &str| -> Vec<usize> {
                std::fs::read_to_string(file(name, "txt"))
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect()
            };
            let x_shape = shape("x");
            assert_eq!(x_shape[2], LATENT);
            Golden {
                x: read(&file("x", "f32")),
                height: x_shape[1] / 2,
                width: x_shape[0] / 2,
                timestep: read(&file("timestep", "f32"))[0],
                context: read(&file("context", "f32")),
                out: read(&file("out", "f32")),
            }
        }
    }

    #[test]
    fn patches_round_trip() {
        let latent: Vec<f32> = (0..LATENT * 4 * 6).map(|v| v as f32).collect();
        let tokens = patchify(&latent, 2, 3);
        assert_eq!(tokens[1], 1.0);
        assert_eq!(tokens[2], 6.0);
        assert_eq!(tokens[4], 24.0);
        assert_eq!(unpatchify(&tokens, 2, 3), latent);
    }

    /// `KUNO_QWEN2512_ENCODER=<qwen_2.5_vl_7b_fp8_scaled> KUNO_QWEN2512_GOLDEN=<dir>`: the
    /// captured prompt rows (cond `step0`, uncond `step1`) against the reference encoder
    /// on the capture script's prompts, with and without the final norm.
    #[test]
    #[ignore = "needs the encoder and captures"]
    fn text_encoder_matches_cpp_engine() {
        use crate::engine::{
            gpu::{Gpu, Select},
            text_encoder::{Prompt, TextEncoder},
            text_gpu::GpuTextEncoder,
            tokenizer::Tokenizer,
        };
        let path = std::env::var("KUNO_QWEN2512_ENCODER").unwrap();
        let encoder = TextEncoder::open(Path::new(&path)).unwrap();
        assert!(encoder.qwen25);
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let on_gpu =
            GpuTextEncoder::new(&gpu, TextEncoder::open(Path::new(&path)).unwrap()).unwrap();
        println!("gpu encoder load {:.1} s", started.elapsed().as_secs_f64());
        let tokenizer = Tokenizer::new().unwrap();
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN2512_GOLDEN").unwrap());
        for (step, text) in [
            (
                0,
                "A red fox sitting on a mossy rock in a misty forest, morning light",
            ),
            (1, "blurry, low quality"),
        ] {
            let full = format!("{TEMPLATE}{text}<|im_end|>\n<|im_start|>assistant\n");
            let ids: Vec<i32> = tokenizer
                .encode(&full)
                .unwrap()
                .into_iter()
                .map(|i| i as i32)
                .collect();
            let prompt = Prompt {
                ids: &ids,
                images: vec![],
            };
            let golden = read(&dir.join(format!("step{step}.context.f32")));
            let started = std::time::Instant::now();
            let cpu = encoder.forward(&prompt).unwrap()[START * CONTEXT..].to_vec();
            let cpu_time = started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            let gpu_rows = on_gpu.forward(&prompt).unwrap()[START * CONTEXT..].to_vec();
            let gpu_time = started.elapsed().as_secs_f64();
            assert_eq!(cpu.len(), golden.len());
            println!(
                "prompt {step}: {} tokens ({} kept): cpu {:.1} dB {cpu_time:.1} s, gpu {:.1} dB {gpu_time:.3} s, gpu from cpu {:.1} dB",
                ids.len(),
                ids.len() - START,
                db(&golden, &cpu),
                db(&golden, &gpu_rows),
                db(&cpu, &gpu_rows)
            );
            assert!(db(&golden, &gpu_rows) > 25.0);
        }
    }

    /// `KUNO_QWEN2512_DIT=<model> KUNO_QWEN2512_GOLDEN=<dir>`: captured C++ steps.
    #[test]
    #[ignore = "needs the model and captures"]
    fn reference_matches_cpp_engine() {
        let mut model =
            Qwen2512Dit::open(Path::new(&std::env::var("KUNO_QWEN2512_DIT").unwrap())).unwrap();
        model.int8 = std::env::var("KUNO_QWEN2512_INT8").is_ok();
        model.attention8 =
            std::env::var("KUNO_QWEN2512_ATTENTION8").map_or(0, |v| v.parse().unwrap());
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN2512_GOLDEN").unwrap());
        for step in 0.. {
            if !dir.join(format!("step{step}.x.f32")).exists() {
                assert!(step > 0);
                break;
            }
            let golden = Golden::load(&dir, step);
            let started = std::time::Instant::now();
            let out = model
                .forward(&Step {
                    x: &golden.x,
                    height: golden.height,
                    width: golden.width,
                    timestep: golden.timestep,
                    context: &golden.context,
                })
                .unwrap();
            let db = db(&golden.out, &out);
            println!(
                "step {step}: t {} {}x{} tokens, {} text, {:.1} s -> {db:.1} dB",
                golden.timestep,
                golden.height,
                golden.width,
                golden.context.len() / CONTEXT,
                started.elapsed().as_secs_f64()
            );
            assert!(db > 30.0, "step {step}: {db:.1} dB");
        }
    }
}
