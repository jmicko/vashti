//! Qwen Image 2.1's image VAE decoder on the CPU: the reference the GPU form is checked
//! against, written from the C++ engine's Wan 2.2-style decoder (`wan_vae.hpp`) for a single
//! image, where every convolution is 2-D (the files' time kernels are 1) and the temporal
//! paths reduce to their first-chunk forms.
//!
//! A 64-channel latent at 1/16 of the image size: un-normalized per channel, a 1x1 conv, a
//! 3x3 conv to 1152 channels, a middle of residual block, single-head attention over all
//! cells and residual block, five up blocks of three residual blocks each (1152, 1152, 576,
//! 288, 144 channels), the first four ending in a nearest 2x upsample with a 3x3 conv plus a
//! channel-repeating depth-to-space shortcut, then RMS norm, SiLU and a 3x3 conv to RGBA.
//! Residual blocks are RMS norm (over channels, times a weight), SiLU, 3x3 conv, twice, plus
//! the input (or its 1x1 projection).

use std::path::Path;

use rayon::prelude::*;

use super::{conv::threads, safetensors::Safetensors};

pub const LATENT: usize = 64;
/// Image channels (RGBA).
pub const OUT: usize = 4;
/// Latent cells per image side.
pub const SCALE: usize = 16;

// The engine's statistics, digit for digit (one entry is near pi / 6 by chance).
#[allow(clippy::approx_constant)]
pub(crate) const MEAN: [f32; LATENT] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510,
    0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391,
    -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342,
    -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680,
    0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162,
    1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
pub(crate) const STD: [f32; LATENT] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904,
    2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075,
    3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490,
    3.7662, 3.6746, 3.4563, 3.8161,
];

/// Decoder widths: the middle, then each up block's output.
pub(super) const WIDTHS: [usize; 6] = [1152, 1152, 1152, 576, 288, 144];
/// Encoder widths: the input conv's, then each down block's output.
pub(super) const ENCODER_WIDTHS: [usize; 6] = [96, 96, 192, 384, 768, 768];
/// Whether down block `i`'s downsample has a temporal factor of 2 (only its shortcut's
/// channel grouping changes for a single image).
pub(super) const ENCODER_TEMPORAL: [bool; 4] = [false, true, true, true];
/// Whether up block `i`'s upsample has a temporal factor of 2 (which only changes how its
/// shortcut groups channels for a single image).
pub(super) const TEMPORAL: [bool; 4] = [true, true, true, false];

/// A feature map `[channels][height][width]`.
#[derive(Clone)]
pub struct Map {
    pub data: Vec<f32>,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}
impl Map {
    fn plane(&self) -> usize {
        self.height * self.width
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// `x / sqrt(mean over channels of x^2 + 1e-12) * gamma[c]` at every cell.
fn rms_norm(x: &Map, gamma: &[f32]) -> Map {
    let plane = x.plane();
    let mut scale = vec![0f32; plane];
    scale.par_iter_mut().enumerate().for_each(|(cell, s)| {
        let mut sum = 0f32;
        for c in 0..x.channels {
            let v = x.data[c * plane + cell];
            sum += v * v;
        }
        *s = 1.0 / (sum / x.channels as f32 + 1e-12).sqrt();
    });
    let mut data = x.data.clone();
    data.par_chunks_exact_mut(plane)
        .zip(gamma)
        .for_each(|(row, g)| {
            for (v, s) in row.iter_mut().zip(&scale) {
                *v *= s * g;
            }
        });
    Map { data, ..*x }
}

/// Stride-1 convolution with zero padding `(k - 1) / 2`: `weight` `[out][in][k][k]`.
fn conv(x: &Map, weight: &[f32], bias: &[f32], k: usize) -> Map {
    conv_strided(x, weight, bias, k, 1, (k - 1) / 2, x.height, x.width)
}
/// Convolution with `stride` reading input `(o * stride + tap - pad)` (zeros outside the map),
/// giving an `out_h` x `out_w` map.
#[allow(clippy::too_many_arguments)]
fn conv_strided(
    x: &Map,
    weight: &[f32],
    bias: &[f32],
    k: usize,
    stride: usize,
    pad: usize,
    out_h: usize,
    out_w: usize,
) -> Map {
    let (cin, h, w) = (x.channels, x.height, x.width);
    let out = bias.len();
    assert_eq!(weight.len(), out * cin * k * k);
    let plane = out_h * out_w;
    let mut data = vec![0f32; out * plane];
    for (o, row) in data.chunks_exact_mut(plane).enumerate() {
        row.fill(bias[o]);
    }
    let mut shifted = vec![0f32; cin * plane];
    let mut tap = vec![0f32; out * cin];
    for dy in 0..k {
        for dx in 0..k {
            // The input under the tap for every output cell, zeros where it leaves the map.
            shifted
                .par_chunks_exact_mut(plane)
                .zip(x.data.par_chunks_exact(h * w))
                .for_each(|(to, from)| {
                    for y in 0..out_h {
                        let sy = (y * stride + dy) as isize - pad as isize;
                        for xx in 0..out_w {
                            let sx = (xx * stride + dx) as isize - pad as isize;
                            to[y * out_w + xx] =
                                if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                                    from[sy as usize * w + sx as usize]
                                } else {
                                    0.0
                                };
                        }
                    }
                });
            for o in 0..out {
                for c in 0..cin {
                    tap[o * cin + c] = weight[((o * cin + c) * k + dy) * k + dx];
                }
            }
            // SAFETY: tap is out x cin, shifted cin x plane, data out x plane, row-major.
            unsafe {
                gemm::gemm(
                    out,
                    plane,
                    cin,
                    data.as_mut_ptr(),
                    1,
                    plane as isize,
                    true,
                    tap.as_ptr(),
                    1,
                    cin as isize,
                    shifted.as_ptr(),
                    1,
                    plane as isize,
                    1.0,
                    1.0,
                    false,
                    false,
                    false,
                    threads(),
                );
            }
        }
    }
    Map {
        data,
        channels: out,
        height: out_h,
        width: out_w,
    }
}

/// The down blocks' shortcut for one image: output channel `o` averages `group` entries
/// `k = o * group + g` of the space-to-depth input, entry `k` being input channel `k / F`
/// at time slot `(k % F) / 4` (slots before `ft - 1` are the zero frame padded in front of a
/// single image) and 2x2 sub-cell `k % 4` (row-major), with `F = 4 ft` and
/// `group = in * F / out`. Without downsampling (`down` false) it is the identity.
pub(super) fn avg_entry(
    o: usize,
    g: usize,
    input: usize,
    output: usize,
    temporal: bool,
) -> Option<(usize, usize)> {
    let ft = if temporal { 2 } else { 1 };
    let factor = 4 * ft;
    let group = input * factor / output;
    let k = o * group + g;
    let (c, r) = (k / factor, k % factor);
    (r / 4 == ft - 1).then_some((c, r % 4))
}
fn avg_down(x: &Map, output: usize, temporal: bool) -> Map {
    let (h, w) = (x.height / 2, x.width / 2);
    let ft = if temporal { 2 } else { 1 };
    let group = x.channels * 4 * ft / output;
    let mut data = vec![0f32; output * h * w];
    data.par_chunks_exact_mut(h * w)
        .enumerate()
        .for_each(|(o, to)| {
            for y in 0..h {
                for xx in 0..w {
                    let mut sum = 0f32;
                    for g in 0..group {
                        if let Some((c, sub)) = avg_entry(o, g, x.channels, output, temporal) {
                            let (i, j) = (sub / 2, sub % 2);
                            sum += x.data[c * x.plane() + (2 * y + i) * x.width + 2 * xx + j];
                        }
                    }
                    to[y * w + xx] = sum / group as f32;
                }
            }
        });
    Map {
        data,
        channels: output,
        height: h,
        width: w,
    }
}

fn add(a: &mut Map, b: &Map) {
    a.data
        .par_iter_mut()
        .zip(&b.data)
        .for_each(|(a, b)| *a += b);
}

/// Nearest-neighbour 2x upsample.
fn upsample(x: &Map) -> Map {
    let (h, w) = (x.height * 2, x.width * 2);
    let mut data = vec![0f32; x.channels * h * w];
    data.par_chunks_exact_mut(h * w)
        .zip(x.data.par_chunks_exact(x.plane()))
        .for_each(|(to, from)| {
            for y in 0..h {
                for xx in 0..w {
                    to[y * w + xx] = from[(y / 2) * x.width + xx / 2];
                }
            }
        });
    Map {
        data,
        channels: x.channels,
        height: h,
        width: w,
    }
}

/// The up blocks' shortcut for one image: each output channel `o` and 2x2 sub-cell `(i, j)`
/// takes input channel `(o * F + (ft - 1) * 4 + i * 2 + j) / repeats`, with `ft` the temporal
/// factor, `F = 4 ft` and `repeats = out * F / in` (channel repetition, then depth-to-space,
/// keeping the last of `ft` frames).
pub(super) fn shortcut_channel(
    o: usize,
    sub: usize,
    input: usize,
    output: usize,
    temporal: bool,
) -> usize {
    let ft = if temporal { 2 } else { 1 };
    let factor = 4 * ft;
    let repeats = output * factor / input;
    (o * factor + (ft - 1) * 4 + sub) / repeats
}
fn dup_up(x: &Map, output: usize, temporal: bool) -> Map {
    let (h, w) = (x.height * 2, x.width * 2);
    let mut data = vec![0f32; output * h * w];
    data.par_chunks_exact_mut(h * w)
        .enumerate()
        .for_each(|(o, to)| {
            for y in 0..h {
                for xx in 0..w {
                    let sub = (y % 2) * 2 + xx % 2;
                    let c = shortcut_channel(o, sub, x.channels, output, temporal);
                    to[y * w + xx] = x.data[c * x.plane() + (y / 2) * x.width + xx / 2];
                }
            }
        });
    Map {
        data,
        channels: output,
        height: h,
        width: w,
    }
}

pub struct QwenVae {
    pub(super) file: Safetensors,
    /// Qwen Image (2512)'s Wan 2.1 VAE: 16 latent channels at 1/8 scale, RGB, causal 3-D
    /// kernels (a single image sees only their last time slice). Decoding only, GPU only.
    pub(super) wan21: bool,
}
/// The Wan 2.1 VAE's latent channels, scale and per-channel statistics (the C++ engine's).
pub const WAN21_LATENT: usize = 16;
pub const WAN21_SCALE: usize = 8;
pub(crate) const WAN21_MEAN: [f32; WAN21_LATENT] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
pub(crate) const WAN21_STD: [f32; WAN21_LATENT] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];
impl QwenVae {
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        file.shape("decoder.conv1.weight")
            .filter(|s| *s == [WIDTHS[0], LATENT, 1, 3, 3])
            .ok_or("model_file_invalid")?;
        Ok(Self { file, wan21: false })
    }
    /// Qwen Image 2512's VAE (`qwen_image_vae.safetensors`).
    pub fn open_wan21(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        file.shape("decoder.conv1.weight")
            .filter(|s| *s == [384, WAN21_LATENT, 3, 3, 3])
            .ok_or("model_file_invalid")?;
        Ok(Self { file, wan21: true })
    }
    fn floats(&self, name: &str) -> Result<Vec<f32>, &'static str> {
        let shape = self
            .file
            .shape(name)
            .ok_or("model_tensor_missing")?
            .to_vec();
        self.file.floats(name, &shape)
    }
    fn conv(&self, x: &Map, name: &str, k: usize) -> Result<Map, &'static str> {
        Ok(conv(
            x,
            &self.floats(&format!("{name}.weight"))?,
            &self.floats(&format!("{name}.bias"))?,
            k,
        ))
    }
    fn norm(&self, x: &Map, name: &str) -> Result<Map, &'static str> {
        Ok(rms_norm(x, &self.floats(&format!("{name}.gamma"))?))
    }
    fn residual(&self, x: Map, name: &str, output: usize) -> Result<Map, &'static str> {
        let skip = if x.channels == output {
            x.clone()
        } else {
            self.conv(&x, &format!("{name}.shortcut"), 1)?
        };
        let mut h = self.norm(&x, &format!("{name}.residual.0"))?;
        h.data.par_iter_mut().for_each(|v| *v = silu(*v));
        let mut h = self.conv(&h, &format!("{name}.residual.2"), 3)?;
        h = self.norm(&h, &format!("{name}.residual.3"))?;
        h.data.par_iter_mut().for_each(|v| *v = silu(*v));
        let mut h = self.conv(&h, &format!("{name}.residual.6"), 3)?;
        add(&mut h, &skip);
        Ok(h)
    }
    fn attention(&self, x: Map, name: &str) -> Result<Map, &'static str> {
        let h = self.norm(&x, &format!("{name}.norm"))?;
        let qkv = self.conv(&h, &format!("{name}.to_qkv"), 1)?;
        let (c, n) = (x.channels, x.plane());
        // Tokens are cells; queries and keys [n][c], values [c][n].
        let transpose = |part: usize| -> Vec<f32> {
            let mut out = vec![0f32; n * c];
            for ch in 0..c {
                for t in 0..n {
                    out[t * c + ch] = qkv.data[(part * c + ch) * n + t];
                }
            }
            out
        };
        let (q, k) = (transpose(0), transpose(1));
        let v = &qkv.data[2 * c * n..3 * c * n];
        let scale = 1.0 / (c as f32).sqrt();
        let mut mixed = vec![0f32; c * n];
        let rows: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|t| {
                let query = &q[t * c..][..c];
                let mut scores: Vec<f32> = (0..n)
                    .map(|s| {
                        query
                            .iter()
                            .zip(&k[s * c..][..c])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let top = scores.iter().fold(f32::NEG_INFINITY, |m, s| m.max(*s));
                let mut total = 0f32;
                for s in &mut scores {
                    *s = (*s - top).exp();
                    total += *s;
                }
                let mut out = vec![0f32; c];
                for (ch, o) in out.iter_mut().enumerate() {
                    let values = &v[ch * n..][..n];
                    *o = scores.iter().zip(values).map(|(w, v)| w * v).sum::<f32>() / total;
                }
                out
            })
            .collect();
        for (t, row) in rows.iter().enumerate() {
            for (ch, value) in row.iter().enumerate() {
                mixed[ch * n + t] = *value;
            }
        }
        let mixed = Map { data: mixed, ..x };
        let mut out = self.conv(&mixed, &format!("{name}.proj"), 1)?;
        add(&mut out, &x);
        Ok(out)
    }

    /// The image `[OUT][16 height][16 width]` in [-1, 1] (before clamping) for a diffusion
    /// latent `[LATENT][height][width]`.
    pub fn decode(
        &self,
        latent: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let plane = height * width;
        if latent.len() != LATENT * plane {
            return Err("invalid_input");
        }
        let mut data = latent.to_vec();
        for (c, row) in data.chunks_exact_mut(plane).enumerate() {
            for v in row {
                *v = *v * STD[c] + MEAN[c];
            }
        }
        let x = Map {
            data,
            channels: LATENT,
            height,
            width,
        };
        let x = self.conv(&x, "conv2", 1)?;
        let x = self.conv(&x, "decoder.conv1", 3)?;
        let x = self.residual(x, "decoder.middle.0", WIDTHS[0])?;
        let x = self.attention(x, "decoder.middle.1")?;
        let mut x = self.residual(x, "decoder.middle.2", WIDTHS[0])?;
        for block in 0..5 {
            let output = WIDTHS[block + 1];
            let input = x.clone();
            for i in 0..3 {
                x = self.residual(
                    x,
                    &format!("decoder.upsamples.{block}.upsamples.{i}"),
                    output,
                )?;
            }
            if block < 4 {
                x = upsample(&x);
                x = self.conv(
                    &x,
                    &format!("decoder.upsamples.{block}.upsamples.3.resample.1"),
                    3,
                )?;
                add(&mut x, &dup_up(&input, output, TEMPORAL[block]));
            }
        }
        let mut x = self.norm(&x, "decoder.head.0")?;
        x.data.par_iter_mut().for_each(|v| *v = silu(*v));
        let x = self.conv(&x, "decoder.head.2", 3)?;
        Ok(x.data)
    }
}

impl QwenVae {
    /// The diffusion latent `[LATENT][height / 16][width / 16]` of an RGBA image (interleaved
    /// bytes): scaled to [-1, 1], encoded, the means kept and normalized per channel.
    pub fn encode(
        &self,
        rgba: &[u8],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, &'static str> {
        if !height.is_multiple_of(SCALE)
            || !width.is_multiple_of(SCALE)
            || rgba.len() != height * width * OUT
            || height == 0
            || width == 0
        {
            return Err("invalid_input");
        }
        let plane = height * width;
        let mut data = vec![0f32; OUT * plane];
        for (cell, pixel) in rgba.chunks_exact(OUT).enumerate() {
            for (c, byte) in pixel.iter().enumerate() {
                data[c * plane + cell] = f32::from(*byte) / 255.0 * 2.0 - 1.0;
            }
        }
        let mut x = Map {
            data,
            channels: OUT,
            height,
            width,
        };
        x = self.conv(&x, "encoder.conv1", 3)?;
        for block in 0..5 {
            let output = ENCODER_WIDTHS[block + 1];
            let input = x.clone();
            let base = format!("encoder.downsamples.{block}.downsamples");
            for i in 0..2 {
                x = self.residual(x, &format!("{base}.{i}"), output)?;
            }
            if block < 4 {
                // Pad right and bottom by one, 3x3 conv with stride 2.
                let name = format!("{base}.2.resample.1");
                let (h, w) = (x.height / 2, x.width / 2);
                x = conv_strided(
                    &x,
                    &self.floats(&format!("{name}.weight"))?,
                    &self.floats(&format!("{name}.bias"))?,
                    3,
                    2,
                    0,
                    h,
                    w,
                );
                add(&mut x, &avg_down(&input, output, ENCODER_TEMPORAL[block]));
            } else {
                add(&mut x, &input);
            }
        }
        let x = self.residual(x, "encoder.middle.0", ENCODER_WIDTHS[5])?;
        let x = self.attention(x, "encoder.middle.1")?;
        let x = self.residual(x, "encoder.middle.2", ENCODER_WIDTHS[5])?;
        let mut x = self.norm(&x, "encoder.head.0")?;
        x.data.par_iter_mut().for_each(|v| *v = silu(*v));
        let x = self.conv(&x, "encoder.head.2", 3)?;
        let x = self.conv(&x, "conv1", 1)?;
        let plane = x.plane();
        let mut latent = x.data[..LATENT * plane].to_vec();
        for (c, row) in latent.chunks_exact_mut(plane).enumerate() {
            for v in row {
                *v = (*v - MEAN[c]) / STD[c];
            }
        }
        Ok(latent)
    }
}

/// `[OUT][h][w]` decoder output in [-1, 1] to interleaved RGBA bytes, as the engine rounds.
/// RGBA bytes (opaque) of an RGB decoder output `[3][height][width]` in [-1, 1].
pub fn to_rgb(image: &[f32], height: usize, width: usize) -> Vec<u8> {
    let plane = height * width;
    let mut out = vec![255u8; plane * 4];
    for (cell, pixel) in out.chunks_exact_mut(4).enumerate() {
        for (c, byte) in pixel[..3].iter_mut().enumerate() {
            let v = ((image[c * plane + cell] + 1.0) * 0.5).clamp(0.0, 1.0);
            *byte = (v * 255.0 + 0.5) as u8;
        }
    }
    out
}
pub fn to_rgba(image: &[f32], height: usize, width: usize) -> Vec<u8> {
    let plane = height * width;
    let mut out = vec![0u8; plane * OUT];
    for (cell, pixel) in out.chunks_exact_mut(OUT).enumerate() {
        for (c, byte) in pixel.iter_mut().enumerate() {
            let v = ((image[c * plane + cell] + 1.0) * 0.5).clamp(0.0, 1.0);
            *byte = if v <= 0.0 {
                0
            } else if v >= 1.0 {
                255
            } else {
                (v * 255.0 + 0.5) as u8
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        super::qwen_dit::tests::{db, read},
        *,
    };

    #[test]
    fn shortcut_groups_channels_like_the_engine() {
        // 1152 -> 1152 with a temporal factor: repeats 8, the second frame's sub-cells.
        assert_eq!(shortcut_channel(0, 0, 1152, 1152, true), 0);
        assert_eq!(
            shortcut_channel(5, 3, 1152, 1152, true),
            (5 * 8 + 4 + 3) / 8
        );
        // 576 -> 288 without: repeats 2.
        assert_eq!(shortcut_channel(7, 2, 576, 288, false), (7 * 4 + 2) / 2);
    }

    /// `KUNO_QWEN_VAE=<qwen_image_2.1_vae_bf16.safetensors> KUNO_QWEN_VAE_GOLDEN=<dir with
    /// latent.f32, image.f32>`: the decoder against the engine's decode of the same latent.
    #[test]
    #[ignore]
    fn decoder_matches_cpp_engine() {
        let vae = QwenVae::open(Path::new(&std::env::var("KUNO_QWEN_VAE").unwrap())).unwrap();
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_VAE_GOLDEN").unwrap());
        let shape = std::fs::read_to_string(dir.join("shape.txt")).unwrap();
        let dims: Vec<usize> = shape
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height) = (dims[0], dims[1]);
        let latent = read(&dir.join("latent.f32"));
        let want = read(&dir.join("image.f32"));
        let started = std::time::Instant::now();
        let got = vae.decode(&latent, height, width).unwrap();
        // The engine's output is already scaled to [0, 1] and clamped.
        let got: Vec<f32> = got
            .iter()
            .map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0))
            .collect();
        let db = db(&want, &got);
        let pixels = to_rgba(
            &got.iter().map(|v| v * 2.0 - 1.0).collect::<Vec<_>>(),
            height * 16,
            width * 16,
        );
        let want_pixels = to_rgba(
            &want.iter().map(|v| v * 2.0 - 1.0).collect::<Vec<_>>(),
            height * 16,
            width * 16,
        );
        let worst = pixels
            .iter()
            .zip(&want_pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        println!(
            "{width}x{height} latent: {db:.1} dB, worst byte {worst}, {:.1} s",
            started.elapsed().as_secs_f64()
        );
        assert!(db > 35.0);
    }

    /// `KUNO_QWEN_VAE=<model> KUNO_QWEN_ENCODE_GOLDEN=<dir with image.f32 ([w][h][4] in 0..1),
    /// latent.f32>`: the encoder against the engine's.
    #[test]
    #[ignore]
    fn encoder_matches_cpp_engine() {
        let vae = QwenVae::open(Path::new(&std::env::var("KUNO_QWEN_VAE").unwrap())).unwrap();
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_ENCODE_GOLDEN").unwrap());
        let shape = std::fs::read_to_string(dir.join("shape.txt")).unwrap();
        let dims: Vec<usize> = shape
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height) = (dims[0], dims[1]);
        let image = read(&dir.join("image.f32"));
        let plane = width * height;
        let mut rgba = vec![0u8; plane * OUT];
        for c in 0..OUT {
            for cell in 0..plane {
                rgba[cell * OUT + c] = (image[c * plane + cell] * 255.0).round() as u8;
            }
        }
        let want = read(&dir.join("latent.f32"));
        let started = std::time::Instant::now();
        let got = vae.encode(&rgba, height, width).unwrap();
        println!(
            "{width}x{height}: {:.1} dB, {:.1} s",
            db(&want, &got),
            started.elapsed().as_secs_f64()
        );
        assert!(db(&want, &got) > 30.0);
    }

    #[test]
    fn average_shortcut_groups_channels_like_the_engine() {
        // 96 -> 96 without a temporal factor: a plain 2x2 average of each channel.
        assert_eq!(avg_entry(5, 3, 96, 96, false), Some((5, 3)));
        // 96 -> 192 with one: even outputs read the zero frame, odd ones channel o / 2.
        assert_eq!(avg_entry(4, 1, 96, 192, true), None);
        assert_eq!(avg_entry(5, 2, 96, 192, true), Some((2, 2)));
    }
}
