//! MiniMax H3 video decoder: a 36-layer transformer that turns one tile of video latents into
//! RGB frames. Each latent position becomes a 4x16x16 block of pixels.
//!
//! This is the CPU form. It streams one layer's weights at a time, so it is the numeric
//! reference for the GPU kernels rather than a production path.
use std::path::Path;

use rayon::prelude::*;

use super::{conv::threads, safetensors::Safetensors};

pub const LATENT_CHANNELS: usize = 24;
/// Output frames per latent frame, and output pixels per latent position on each side.
pub const PATCH_FRAMES: usize = 4;
pub const PATCH: usize = 16;
const DIM: usize = 2048;
const HEADS: usize = 32;
const HEAD: usize = DIM / HEADS;
const LAYERS: usize = 36;
const INNER: usize = DIM * 4;
const REGISTERS: usize = 4;
/// Tokens appended after the latent positions: the registers and one zero token.
const EXTRA: usize = REGISTERS + 1;
/// Rotary pairs per axis (time, height, width). The last 16 of 64 head channels stay fixed.
const AXIS_PAIRS: usize = 8;
const PAIRS: usize = AXIS_PAIRS * 3;
const PIXELS: usize = 3 * PATCH_FRAMES * PATCH * PATCH;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct VideoDecoder {
    file: Safetensors,
}
impl VideoDecoder {
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        file.metadata("minimax_h3_video_vae")
            .ok_or("model_file_invalid")?;
        Ok(Self { file })
    }
    /// `latent` is `[24][frames][height][width]` in VAE scale (not diffusion scale). Returns
    /// RGB in 0..=1 as `[3][frames * 4][height * 16][width * 16]`.
    pub fn decode_tile(
        &self,
        latent: &[f32],
        width: usize,
        height: usize,
        frames: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let positions = width * height * frames;
        if positions == 0 || latent.len() != positions * LATENT_CHANNELS {
            return Err("video_latent_invalid");
        }
        let f = &self.file;
        let mut tokens = vec![0f32; positions * LATENT_CHANNELS];
        for (channel, plane) in latent.chunks_exact(positions).enumerate() {
            for (position, value) in plane.iter().enumerate() {
                tokens[position * LATENT_CHANNELS + channel] = *value;
            }
        }
        let tokens = linear(
            &tokens,
            LATENT_CHANNELS,
            &f.f16("post_quant_conv.weight", &[24, 24, 1, 1, 1])?,
            &f.f16("post_quant_conv.bias", &[24])?,
        );
        let mut x = linear(
            &tokens,
            LATENT_CHANNELS,
            &f.f16("decoder.x_embedder.weight", &[DIM, 24])?,
            &f.f16("decoder.x_embedder.bias", &[DIM])?,
        );
        x.extend(f.f16("decoder.register_tokens", &[1, REGISTERS, DIM])?);
        x.resize((positions + EXTRA) * DIM, 0.0);
        let rotary = rotary(width, height, frames);
        for layer in 0..LAYERS {
            let name = |part: &str| format!("decoder.transformer_blocks.{layer}.{part}");
            let mut normed = x.clone();
            rms_norm(
                &mut normed,
                DIM,
                Some(&f.f16(&name("norm1.weight"), &[DIM])?),
            );
            let qkv = linear(
                &normed,
                DIM,
                &f.f16(&name("attn.to_qkv.weight"), &[DIM * 3, DIM])?,
                &f.f16(&name("attn.to_qkv.bias"), &[DIM * 3])?,
            );
            let attended = linear(
                &attention(&qkv, &rotary),
                DIM,
                &f.f16(&name("attn.to_out.weight"), &[DIM, DIM])?,
                &f.f16(&name("attn.to_out.bias"), &[DIM])?,
            );
            add_scaled(&mut x, &attended, &f.f16(&name("scale1"), &[DIM])?);

            let mut normed = x.clone();
            rms_norm(
                &mut normed,
                DIM,
                Some(&f.f16(&name("norm2.weight"), &[DIM])?),
            );
            let gates = linear(
                &normed,
                DIM,
                &f.f16(&name("ff.w1.weight"), &[INNER * 2, DIM])?,
                &f.f16(&name("ff.w1.bias"), &[INNER * 2])?,
            );
            let mut gated = vec![0f32; gates.len() / 2];
            gated
                .par_chunks_exact_mut(INNER)
                .zip(gates.par_chunks_exact(INNER * 2))
                .for_each(|(out, gates)| {
                    let (gate, value) = gates.split_at(INNER);
                    for ((out, gate), value) in out.iter_mut().zip(gate).zip(value) {
                        *out = gate / (1.0 + (-gate).exp()) * value;
                    }
                });
            let fed = linear(
                &gated,
                INNER,
                &f.f16(&name("ff.w2.weight"), &[DIM, INNER])?,
                &f.f16(&name("ff.w2.bias"), &[DIM])?,
            );
            add_scaled(&mut x, &fed, &f.f16(&name("scale2"), &[DIM])?);
        }
        x.truncate(positions * DIM);
        layer_norm(
            &mut x,
            &f.f16("decoder.norm_out.weight", &[DIM])?,
            &f.f16("decoder.norm_out.bias", &[DIM])?,
        );
        let patches = linear(
            &x,
            DIM,
            &f.f16("decoder.proj_out.weight", &[PIXELS, DIM])?,
            &f.f16("decoder.proj_out.bias", &[PIXELS])?,
        );
        Ok(unpatchify(&patches, width, height, frames))
    }
}
/// `x` is `[rows][inputs]`, `weight` is `[outputs][inputs]`; returns `[rows][outputs]`.
fn linear(x: &[f32], inputs: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let (rows, outputs) = (x.len() / inputs, bias.len());
    assert!(x.len() == rows * inputs && weight.len() == outputs * inputs);
    let mut out = vec![0f32; rows * outputs];
    for row in out.chunks_exact_mut(outputs) {
        row.copy_from_slice(bias);
    }
    // SAFETY: the three matrices span exactly the slices checked above; the right-hand side
    // is the weight read transposed (row stride 1, column stride `inputs`).
    unsafe {
        gemm::gemm(
            rows,
            outputs,
            inputs,
            out.as_mut_ptr(),
            1,
            outputs as isize,
            true,
            x.as_ptr(),
            1,
            inputs as isize,
            weight.as_ptr(),
            inputs as isize,
            1,
            1.0,
            1.0,
            false,
            false,
            false,
            threads(),
        );
    }
    out
}
fn rms_norm(x: &mut [f32], width: usize, weight: Option<&[f32]>) {
    x.par_chunks_exact_mut(width).for_each(|row| {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let scale = 1.0 / (mean + 1e-5).sqrt();
        match weight {
            Some(weight) => row
                .iter_mut()
                .zip(weight)
                .for_each(|(v, w)| *v *= scale * w),
            None => row.iter_mut().for_each(|v| *v *= scale),
        }
    });
}
fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32]) {
    x.par_chunks_exact_mut(DIM).for_each(|row| {
        let mean = row.iter().sum::<f32>() / DIM as f32;
        let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / DIM as f32;
        let scale = 1.0 / (variance + 1e-5).sqrt();
        for ((v, w), b) in row.iter_mut().zip(weight).zip(bias) {
            *v = (*v - mean) * scale * w + b;
        }
    });
}
fn add_scaled(x: &mut [f32], update: &[f32], scale: &[f32]) {
    x.par_chunks_exact_mut(DIM)
        .zip(update.par_chunks_exact(DIM))
        .for_each(|(x, update)| {
            for ((x, update), scale) in x.iter_mut().zip(update).zip(scale) {
                *x += update * scale;
            }
        });
}
/// Cosine and sine per token and rotary pair: `[tokens][24][2]`. Positions are centred and
/// span -2*pi..2*pi on each axis; the appended tokens sit at zero (no rotation).
fn rotary(width: usize, height: usize, frames: usize) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity((width * height * frames + EXTRA) * PAIRS);
    let place = |index: usize, count: usize| {
        (2.0 * ((index as f64 + 0.5) / count as f64) - 1.0) * std::f64::consts::TAU
    };
    for t in 0..frames {
        for h in 0..height {
            for w in 0..width {
                for position in [place(t, frames), place(h, height), place(w, width)] {
                    for pair in 0..AXIS_PAIRS {
                        let angle = position * 100f64.powf(-(pair as f64) / AXIS_PAIRS as f64);
                        out.push([angle.cos() as f32, angle.sin() as f32]);
                    }
                }
            }
        }
    }
    out.resize(out.len() + EXTRA * PAIRS, [1.0, 0.0]);
    out
}
/// `qkv` is `[tokens][heads][query 64, key 64, value 64]`; returns `[tokens][heads][64]`.
fn attention(qkv: &[f32], rotary: &[[f32; 2]]) -> Vec<f32> {
    let tokens = qkv.len() / (DIM * 3);
    let heads: Vec<Vec<f32>> = (0..HEADS)
        .into_par_iter()
        .map(|head| {
            let mut query = vec![0f32; tokens * HEAD];
            let mut key = vec![0f32; tokens * HEAD];
            let mut value = vec![0f32; tokens * HEAD];
            for token in 0..tokens {
                let from = &qkv[(token * HEADS + head) * HEAD * 3..][..HEAD * 3];
                let row = token * HEAD..(token + 1) * HEAD;
                query[row.clone()].copy_from_slice(&from[..HEAD]);
                key[row.clone()].copy_from_slice(&from[HEAD..HEAD * 2]);
                value[row].copy_from_slice(&from[HEAD * 2..]);
            }
            for side in [&mut query, &mut key] {
                for (row, rotary) in side.chunks_exact_mut(HEAD).zip(rotary.chunks_exact(PAIRS)) {
                    let mean = row.iter().map(|v| v * v).sum::<f32>() / HEAD as f32;
                    let scale = 1.0 / (mean + 1e-5).sqrt();
                    row.iter_mut().for_each(|v| *v *= scale);
                    // Pair `i` is channels `i` and `i + 24`.
                    for (pair, [cos, sin]) in rotary.iter().enumerate() {
                        let (a, b) = (row[pair], row[pair + PAIRS]);
                        row[pair] = cos * a - sin * b;
                        row[pair + PAIRS] = sin * a + cos * b;
                    }
                }
            }
            let mut scores = vec![0f32; tokens * tokens];
            // SAFETY: query and key are `tokens x 64`, scores is `tokens x tokens`; the key
            // is read transposed.
            unsafe {
                gemm::gemm(
                    tokens,
                    tokens,
                    HEAD,
                    scores.as_mut_ptr(),
                    1,
                    tokens as isize,
                    false,
                    query.as_ptr(),
                    1,
                    HEAD as isize,
                    key.as_ptr(),
                    HEAD as isize,
                    1,
                    0.0,
                    1.0 / (HEAD as f32).sqrt(),
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            for row in scores.chunks_exact_mut(tokens) {
                let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - top).exp();
                    sum += *v;
                }
                let scale = 1.0 / sum;
                row.iter_mut().for_each(|v| *v *= scale);
            }
            let mut out = vec![0f32; tokens * HEAD];
            // SAFETY: scores is `tokens x tokens`, value and out are `tokens x 64`.
            unsafe {
                gemm::gemm(
                    tokens,
                    HEAD,
                    tokens,
                    out.as_mut_ptr(),
                    1,
                    HEAD as isize,
                    false,
                    scores.as_ptr(),
                    1,
                    tokens as isize,
                    value.as_ptr(),
                    1,
                    HEAD as isize,
                    0.0,
                    1.0,
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            out
        })
        .collect();
    let mut out = vec![0f32; tokens * DIM];
    for (head, values) in heads.iter().enumerate() {
        for (token, row) in values.chunks_exact(HEAD).enumerate() {
            out[(token * HEADS + head) * HEAD..][..HEAD].copy_from_slice(row);
        }
    }
    out
}
/// `patches` is `[positions][3][4][16][16]`; scatters blocks into frames and applies the
/// pixel mean and deviation.
fn unpatchify(patches: &[f32], width: usize, height: usize, frames: usize) -> Vec<f32> {
    let (out_w, out_h, out_t) = (width * PATCH, height * PATCH, frames * PATCH_FRAMES);
    let plane = out_w * out_h * out_t;
    let mut out = vec![0f32; 3 * plane];
    for (position, patch) in patches.chunks_exact(PIXELS).enumerate() {
        let (w, h, t) = (
            position % width,
            position / width % height,
            position / (width * height),
        );
        for (index, value) in patch.iter().enumerate() {
            let (channel, dt, dh, dw) = (
                index / (PATCH_FRAMES * PATCH * PATCH),
                index / (PATCH * PATCH) % PATCH_FRAMES,
                index / PATCH % PATCH,
                index % PATCH,
            );
            let at = ((t * PATCH_FRAMES + dt) * out_h + h * PATCH + dh) * out_w + w * PATCH + dw;
            out[channel * plane + at] = (value * STD[channel] + MEAN[channel]).clamp(0.0, 1.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slow double-precision decoder written from the layer definitions. It shares no code
    /// with the fast path and is the arbiter between it and the C++ engine.
    mod exact {
        use super::super::*;

        fn dot(x: &[f64], weight: &[f32]) -> f64 {
            let mut sums = [0f64; 8];
            let (xs, ws) = (x.chunks_exact(8), weight.chunks_exact(8));
            let tail: f64 = xs
                .remainder()
                .iter()
                .zip(ws.remainder())
                .map(|(x, w)| x * f64::from(*w))
                .sum();
            for (x, w) in xs.zip(ws) {
                for i in 0..8 {
                    sums[i] += x[i] * f64::from(w[i]);
                }
            }
            sums.iter().sum::<f64>() + tail
        }
        fn project(x: &[f64], inputs: usize, weight: &[f32], bias: &[f32]) -> Vec<f64> {
            x.par_chunks_exact(inputs)
                .flat_map_iter(|row| {
                    weight
                        .chunks_exact(inputs)
                        .zip(bias)
                        .map(move |(weights, bias)| dot(row, weights) + f64::from(*bias))
                })
                .collect()
        }
        fn unit_rms(row: &mut [f64]) {
            let mean = row.iter().map(|v| v * v).sum::<f64>() / row.len() as f64;
            let root = (mean + 1e-5).sqrt();
            row.iter_mut().for_each(|v| *v /= root);
        }
        fn turn(row: &mut [f64], place: [f64; 3]) {
            for (axis, position) in place.into_iter().enumerate() {
                for step in 0..8 {
                    let angle = position * (-(step as f64) / 8.0 * 100f64.ln()).exp();
                    let (low, high) = (axis * 8 + step, axis * 8 + step + 24);
                    let (a, b) = (row[low], row[high]);
                    row[low] = a * angle.cos() - b * angle.sin();
                    row[high] = a * angle.sin() + b * angle.cos();
                }
            }
        }
        pub fn decode_tile(
            file: &Safetensors,
            latent: &[f32],
            width: usize,
            height: usize,
            frames: usize,
        ) -> Vec<f64> {
            let get = |name: &str, shape: &[usize]| file.f16(name, shape).unwrap();
            let positions = width * height * frames;
            let tokens = positions + 5;
            let centre = |index: usize, count: usize| {
                (2.0 * (index as f64 + 0.5) / count as f64 - 1.0) * 2.0 * std::f64::consts::PI
            };
            let places: Vec<[f64; 3]> = (0..tokens)
                .map(|token| {
                    if token >= positions {
                        return [0.0; 3];
                    }
                    [
                        centre(token / (width * height), frames),
                        centre(token / width % height, height),
                        centre(token % width, width),
                    ]
                })
                .collect();
            let mut z = vec![0f64; positions * 24];
            for position in 0..positions {
                for channel in 0..24 {
                    z[position * 24 + channel] = f64::from(latent[channel * positions + position]);
                }
            }
            let z = project(
                &z,
                24,
                &get("post_quant_conv.weight", &[24, 24, 1, 1, 1]),
                &get("post_quant_conv.bias", &[24]),
            );
            let mut x = project(
                &z,
                24,
                &get("decoder.x_embedder.weight", &[2048, 24]),
                &get("decoder.x_embedder.bias", &[2048]),
            );
            x.extend(
                get("decoder.register_tokens", &[1, 4, 2048])
                    .into_iter()
                    .map(f64::from),
            );
            x.extend(std::iter::repeat_n(0.0, 2048));
            for layer in 0..36 {
                let name = |part: &str| format!("decoder.transformer_blocks.{layer}.{part}");
                let weighted_norm = |x: &[f64], weight: &[f32]| {
                    let mut out = x.to_vec();
                    for row in out.chunks_exact_mut(2048) {
                        unit_rms(row);
                        row.iter_mut()
                            .zip(weight)
                            .for_each(|(v, w)| *v *= f64::from(*w));
                    }
                    out
                };
                let qkv = project(
                    &weighted_norm(&x, &get(&name("norm1.weight"), &[2048])),
                    2048,
                    &get(&name("attn.to_qkv.weight"), &[6144, 2048]),
                    &get(&name("attn.to_qkv.bias"), &[6144]),
                );
                // Per token and head: 64 query, 64 key, 64 value channels.
                let part = |token: usize, head: usize, which: usize| {
                    let start = token * 6144 + head * 192 + which * 64;
                    let mut row = qkv[start..start + 64].to_vec();
                    if which < 2 {
                        unit_rms(&mut row);
                        turn(&mut row, places[token]);
                    }
                    row
                };
                let mixed: Vec<f64> = (0..tokens * 32)
                    .into_par_iter()
                    .flat_map_iter(|index| {
                        let (token, head) = (index / 32, index % 32);
                        let query = part(token, head, 0);
                        let scores: Vec<f64> = (0..tokens)
                            .map(|other| {
                                let key = part(other, head, 1);
                                query.iter().zip(&key).map(|(q, k)| q * k).sum::<f64>() / 8.0
                            })
                            .collect();
                        let top = scores.iter().copied().fold(f64::MIN, f64::max);
                        let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                        let total: f64 = weights.iter().sum();
                        let mut out = vec![0f64; 64];
                        for (other, weight) in weights.iter().enumerate() {
                            let start = other * 6144 + head * 192 + 128;
                            for (out, value) in out.iter_mut().zip(&qkv[start..start + 64]) {
                                *out += weight / total * value;
                            }
                        }
                        out
                    })
                    .collect();
                let attended = project(
                    &mixed,
                    2048,
                    &get(&name("attn.to_out.weight"), &[2048, 2048]),
                    &get(&name("attn.to_out.bias"), &[2048]),
                );
                let scale = get(&name("scale1"), &[2048]);
                for (index, value) in attended.iter().enumerate() {
                    x[index] += value * f64::from(scale[index % 2048]);
                }
                let gates = project(
                    &weighted_norm(&x, &get(&name("norm2.weight"), &[2048])),
                    2048,
                    &get(&name("ff.w1.weight"), &[16384, 2048]),
                    &get(&name("ff.w1.bias"), &[16384]),
                );
                let gated: Vec<f64> = gates
                    .chunks_exact(16384)
                    .flat_map(|row| {
                        (0..8192).map(|i| row[i] / (1.0 + (-row[i]).exp()) * row[i + 8192])
                    })
                    .collect();
                let fed = project(
                    &gated,
                    8192,
                    &get(&name("ff.w2.weight"), &[2048, 8192]),
                    &get(&name("ff.w2.bias"), &[2048]),
                );
                let scale = get(&name("scale2"), &[2048]);
                for (index, value) in fed.iter().enumerate() {
                    x[index] += value * f64::from(scale[index % 2048]);
                }
            }
            let (weight, bias) = (
                get("decoder.norm_out.weight", &[2048]),
                get("decoder.norm_out.bias", &[2048]),
            );
            x.truncate(positions * 2048);
            for row in x.chunks_exact_mut(2048) {
                let mean = row.iter().sum::<f64>() / 2048.0;
                let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 2048.0;
                for (index, value) in row.iter_mut().enumerate() {
                    *value = (*value - mean) / (variance + 1e-5).sqrt() * f64::from(weight[index])
                        + f64::from(bias[index]);
                }
            }
            let patches = project(
                &x,
                2048,
                &get("decoder.proj_out.weight", &[3072, 2048]),
                &get("decoder.proj_out.bias", &[3072]),
            );
            let (out_w, out_h, out_t) = (width * 16, height * 16, frames * 4);
            let mut out = vec![0f64; 3 * out_t * out_h * out_w];
            for channel in 0..3 {
                for t in 0..out_t {
                    for h in 0..out_h {
                        for w in 0..out_w {
                            let position = (t / 4 * height + h / 16) * width + w / 16;
                            let inside = ((channel * 4 + t % 4) * 16 + h % 16) * 16 + w % 16;
                            let value = patches[position * 3072 + inside] * f64::from(STD[channel])
                                + f64::from(MEAN[channel]);
                            out[((channel * out_t + t) * out_h + h) * out_w + w] =
                                value.clamp(0.0, 1.0);
                        }
                    }
                }
            }
            out
        }
    }

    fn floats(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }
    /// Signal-to-error ratio in dB of `got` against the exact pixels.
    fn decibels(exact: &[f64], got: &[f32]) -> f64 {
        assert_eq!(exact.len(), got.len());
        let signal: f64 = exact.iter().map(|v| v * v).sum();
        let error: f64 = exact
            .iter()
            .zip(got)
            .map(|(e, g)| (e - f64::from(*g)).powi(2))
            .sum();
        10.0 * (signal / error.max(f64::MIN_POSITIVE)).log10()
    }

    #[test]
    fn rotary_positions_are_centred_and_extras_do_not_rotate() {
        let table = rotary(2, 1, 1);
        assert_eq!(table.len(), (2 + EXTRA) * PAIRS);
        // One frame, one row: both sit at angle zero on the first pair of those axes.
        assert_eq!(table[0], [1.0, 0.0]);
        assert_eq!(table[AXIS_PAIRS], [1.0, 0.0]);
        // Width 2: positions -pi and +pi, so the first width pair is (-1, 0).
        assert!((table[AXIS_PAIRS * 2][0] + 1.0).abs() < 1e-6);
        assert!((table[PAIRS + AXIS_PAIRS * 2][0] + 1.0).abs() < 1e-6);
        assert!(table[2 * PAIRS..].iter().all(|v| *v == [1.0, 0.0]));
    }
    #[test]
    fn unpatchify_places_every_block() {
        let (width, height, frames) = (2, 3, 2);
        let patches: Vec<f32> = (0..width * height * frames * PIXELS)
            .map(|i| ((i % 251) as f32 / 251.0 - MEAN[0]) / STD[0])
            .collect();
        let out = unpatchify(&patches, width, height, frames);
        let (out_w, out_h, out_t) = (width * PATCH, height * PATCH, frames * PATCH_FRAMES);
        for (t, h, w) in [(0, 0, 0), (5, 33, 17), (7, 47, 31), (3, 16, 15)] {
            let position = (t / 4 * height + h / 16) * width + w / 16;
            let inside = (t % 4 * 16 + h % 16) * 16 + w % 16;
            let want = ((position * PIXELS + inside) % 251) as f32 / 251.0;
            let got = out[(t * out_h + h) * out_w + w];
            assert!((got - want).abs() < 1e-6, "{t} {h} {w}: {got} vs {want}");
        }
        assert_eq!(out.len(), 3 * out_t * out_h * out_w);
    }
    /// Needs the model file and a tile captured from the C++ engine:
    /// `KUNO_VIDEO_VAE=<model> KUNO_VIDEO_GOLDEN=<dir>` with `tile-latent.f32`,
    /// `tile-pixels.f32` and `tile-shape.txt` ("width height frames").
    #[test]
    #[ignore]
    fn tile_matches_exact_reference_and_cpp_engine() {
        let model = std::env::var("KUNO_VIDEO_VAE").unwrap();
        let golden = std::path::PathBuf::from(std::env::var("KUNO_VIDEO_GOLDEN").unwrap());
        let shape: Vec<usize> = std::fs::read_to_string(golden.join("tile-shape.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height, frames) = (shape[0], shape[1], shape[2]);
        let latent = floats(&golden.join("tile-latent.f32"));
        let cpp = floats(&golden.join("tile-pixels.f32"));
        let decoder = VideoDecoder::open(Path::new(&model)).unwrap();
        let started = std::time::Instant::now();
        let rust = decoder.decode_tile(&latent, width, height, frames).unwrap();
        let rust_seconds = started.elapsed().as_secs_f64();
        let rust_vs_cpp = decibels(
            &cpp.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
            &rust,
        );
        println!(
            "tile {width}x{height}x{frames}: rust {rust_seconds:.1} s, rust vs c++ {rust_vs_cpp:.1} dB"
        );
        let started = std::time::Instant::now();
        let exact = exact::decode_tile(&decoder.file, &latent, width, height, frames);
        println!("exact {:.0} s", started.elapsed().as_secs_f64());
        let (rust_db, cpp_db) = (decibels(&exact, &rust), decibels(&exact, &cpp));
        println!("rust vs exact {rust_db:.1} dB, c++ vs exact {cpp_db:.1} dB");
        assert!(rust_db > 80.0 && rust_db >= cpp_db);
    }
}
