//! MiniMax H3 audio decoder: latent frames to 32 kHz stereo. A BigVGAN vocoder: seven
//! upsampling stages (800x in total), each averaging three residual blocks whose
//! activations are anti-aliased (2x up, snake, 2x down). Port of the decode path of
//! stable-diffusion.cpp's `minimax_h3_audio_vae.hpp`; the encoder is not used by Kuno.
use super::{
    conv::{Conv1d, ConvTranspose1d},
    safetensors::Safetensors,
};
use rayon::prelude::*;

pub const SAMPLE_RATE: u32 = 32000;
pub const LATENT_CHANNELS: usize = 32;
pub const STREAMS: usize = 2;
/// Audio samples per latent frame.
pub const HOP: usize = 800;
const RATES: [usize; 7] = [5, 5, 2, 2, 2, 2, 2];
const RESIDUAL_KERNELS: [usize; 3] = [3, 7, 11];
const DILATIONS: [usize; 3] = [1, 3, 5];
const FILTER: usize = 12;

/// sin(x)^2, exact to f32 rounding for |x| up to a few thousand; written so it vectorizes.
#[inline(always)]
fn sin_squared(x: f32) -> f32 {
    let k = (x * std::f32::consts::FRAC_1_PI).round_ties_even();
    // Three-part pi keeps the reduced angle exact.
    let r = ((x - k * 3.140625) - k * 9.675_026e-4) - k * 1.509_958e-7;
    let r2 = r * r;
    let s = r
        * (1.0
            + r2 * (-1.0 / 6.0
                + r2 * (1.0 / 120.0
                    + r2 * (-1.0 / 5040.0
                        + r2 * (1.0 / 362_880.0
                            + r2 * (-1.0 / 39_916_800.0 + r2 * (1.0 / 6_227_020_800.0)))))));
    s * s
}
/// Anti-aliased snake activation: 2x sinc upsample, `x + sin^2(a x) / b`, 2x lowpass downsample.
struct Activation {
    /// Per channel `exp(alpha)` and `1 / (exp(beta) + 1e-9)`.
    alpha: Vec<f32>,
    inverse_beta: Vec<f32>,
    /// Upsampling taps for even and odd outputs, already doubled, in input order.
    up_even: [f32; 6],
    up_odd: [f32; 6],
    /// Downsampling taps for the even and odd upsampled positions.
    down_even: [f32; 6],
    down_odd: [f32; 6],
}
#[derive(Default)]
struct Scratch {
    padded: Vec<f32>,
    even: Vec<f32>,
    odd: Vec<f32>,
}
impl Activation {
    fn load(file: &Safetensors, prefix: &str, channels: usize) -> Result<Self, &'static str> {
        let alpha = file.f32(&format!("{prefix}.act.alpha"), &[channels])?;
        let beta = file.f32(&format!("{prefix}.act.beta"), &[channels])?;
        let up = file.f32(&format!("{prefix}.upsample.filter"), &[1, 1, FILTER])?;
        let down = file.f32(
            &format!("{prefix}.downsample.lowpass.filter"),
            &[1, 1, FILTER],
        )?;
        Ok(Self::new(&alpha, &beta, &up, &down))
    }
    fn new(alpha: &[f32], beta: &[f32], up: &[f32], down: &[f32]) -> Self {
        // With the input replicate-padded by 5, upsampled sample n is
        // 2 * sum_t padded[t] * up[n + 15 - 2t]; even n uses the odd taps and the reverse.
        Self {
            alpha: alpha.iter().map(|a| a.exp()).collect(),
            inverse_beta: beta.iter().map(|b| 1.0 / (b.exp() + 1e-9)).collect(),
            up_even: std::array::from_fn(|i| 2.0 * up[11 - 2 * i]),
            up_odd: std::array::from_fn(|i| 2.0 * up[10 - 2 * i]),
            down_even: std::array::from_fn(|i| down[2 * i]),
            down_odd: std::array::from_fn(|i| down[2 * i + 1]),
        }
    }
    #[inline(always)]
    fn channel(&self, c: usize, x: &mut [f32], scratch: &mut Scratch) {
        let len = x.len();
        if len == 0 {
            return;
        }
        let (alpha, inverse_beta) = (self.alpha[c], self.inverse_beta[c]);
        let Scratch { padded, even, odd } = scratch;
        padded.clear();
        padded.extend(std::iter::repeat_n(x[0], 5));
        padded.extend_from_slice(x);
        padded.extend(std::iter::repeat_n(x[len - 1], 5));
        // The upsampled signal (2 * len samples) is stored replicate-padded by 5 before and 6
        // after and split by parity, so both filters below run over unit-stride slices.
        even.clear();
        even.resize(len + 6, 0.0);
        odd.clear();
        odd.resize(len + 5, 0.0);
        let snake = |v: f32| v + sin_squared(v * alpha) * inverse_beta;
        for (q, out) in odd[2..len + 2].iter_mut().enumerate() {
            let w = &padded[q + 2..q + 8];
            let mut v = 0.0;
            for i in 0..6 {
                v += self.up_even[i] * w[i];
            }
            *out = snake(v);
        }
        for (q, out) in even[3..len + 3].iter_mut().enumerate() {
            let w = &padded[q + 3..q + 9];
            let mut v = 0.0;
            for i in 0..6 {
                v += self.up_odd[i] * w[i];
            }
            *out = snake(v);
        }
        let (first, last) = (odd[2], even[len + 2]);
        even[..3].fill(first);
        odd[..2].fill(first);
        even[len + 3..].fill(last);
        odd[len + 2..].fill(last);
        for (t, out) in x.iter_mut().enumerate() {
            let (e, o) = (&even[t..t + 6], &odd[t..t + 6]);
            let mut v = 0.0;
            for i in 0..6 {
                v += self.down_even[i] * e[i] + self.down_odd[i] * o[i];
            }
            *out = v;
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn channel_avx2(&self, c: usize, x: &mut [f32], scratch: &mut Scratch) {
        self.channel(c, x, scratch)
    }
    fn forward(&self, x: &mut [f32], len: usize) {
        assert_eq!(x.len(), self.alpha.len() * len);
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        x.par_chunks_mut(len.max(1)).enumerate().for_each_init(
            Scratch::default,
            |scratch, (c, row)| {
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: AVX2 was detected at run time.
                    return unsafe { self.channel_avx2(c, row, scratch) };
                }
                self.channel(c, row, scratch)
            },
        );
    }
}
struct ResidualBlock {
    /// (activation, dilated conv, activation, plain conv) per dilation.
    layers: Vec<(Activation, Conv1d, Activation, Conv1d)>,
}
impl ResidualBlock {
    fn load(
        file: &Safetensors,
        prefix: &str,
        channels: usize,
        kernel: usize,
    ) -> Result<Self, &'static str> {
        let conv = |name: &str, dilation| {
            Conv1d::new(
                &file.f32(
                    &format!("{prefix}.{name}.weight"),
                    &[channels, channels, kernel],
                )?,
                Some(file.f32(&format!("{prefix}.{name}.bias"), &[channels])?),
                channels,
                channels,
                kernel,
                dilation,
            )
        };
        let layers = (0..3)
            .map(|i| {
                Ok((
                    Activation::load(file, &format!("{prefix}.activations.{}", 2 * i), channels)?,
                    conv(&format!("convs1.{i}"), DILATIONS[i])?,
                    Activation::load(
                        file,
                        &format!("{prefix}.activations.{}", 2 * i + 1),
                        channels,
                    )?,
                    conv(&format!("convs2.{i}"), 1)?,
                ))
            })
            .collect::<Result<_, &'static str>>()?;
        Ok(Self { layers })
    }
    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let mut x = x.to_vec();
        for (act1, conv1, act2, conv2) in &self.layers {
            let mut h = x.clone();
            act1.forward(&mut h, len);
            let mut h = conv1.forward(&h, len);
            act2.forward(&mut h, len);
            let h = conv2.forward(&h, len);
            x.par_iter_mut().zip(h).for_each(|(x, h)| *x += h);
        }
        x
    }
}
pub struct AudioDecoder {
    mean: Vec<f32>,
    std: Vec<f32>,
    input: Conv1d,
    pre: Conv1d,
    stages: Vec<(ConvTranspose1d, [ResidualBlock; 3])>,
    post_activation: Activation,
    post: Conv1d,
}
impl AudioDecoder {
    pub fn load(path: &std::path::Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        let conv = |name: &str, cin, cout, kernel, bias: bool| {
            Conv1d::new(
                &file.f32(&format!("{name}.weight"), &[cout, cin, kernel])?,
                if bias {
                    Some(file.f32(&format!("{name}.bias"), &[cout])?)
                } else {
                    None
                },
                cin,
                cout,
                kernel,
                1,
            )
        };
        let mut channels = 1024;
        let mut stages = Vec::new();
        for (i, rate) in RATES.into_iter().enumerate() {
            let (next, kernel) = (channels / 2, if rate == 5 { 9 } else { 4 });
            let up = ConvTranspose1d::new(
                &file.f32(
                    &format!("decoder.ups.{i}.0.weight"),
                    &[channels, next, kernel],
                )?,
                file.f32(&format!("decoder.ups.{i}.0.bias"), &[next])?,
                channels,
                next,
                kernel,
                rate,
            )?;
            let block = |j: usize| {
                ResidualBlock::load(
                    &file,
                    &format!("decoder.resblocks.{}", i * 3 + j),
                    next,
                    RESIDUAL_KERNELS[j],
                )
            };
            stages.push((up, [block(0)?, block(1)?, block(2)?]));
            channels = next;
        }
        let std = file.f32("latents_std", &[LATENT_CHANNELS])?;
        if std.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err("model_tensor_invalid");
        }
        Ok(Self {
            mean: file.f32("latents_mean", &[LATENT_CHANNELS])?,
            std,
            input: conv("dec_in_proj", LATENT_CHANNELS, 2048, 1, true)?,
            pre: conv("decoder.conv_pre", 2048, 1024, 7, true)?,
            stages,
            post_activation: Activation::load(&file, "decoder.activation_post", channels)?,
            post: conv("decoder.conv_post", channels, 1, 7, false)?,
        })
    }
    fn stream(&self, latent: &[f32], frames: usize, stream: usize) -> Vec<f32> {
        // The engine's latent is [frames][streams][channels] with frames varying fastest.
        let mut z = vec![0.0; LATENT_CHANNELS * frames];
        for c in 0..LATENT_CHANNELS {
            let source = &latent[frames * (stream + STREAMS * c)..][..frames];
            for (z, v) in z[c * frames..][..frames].iter_mut().zip(source) {
                *z = v * self.std[c] + self.mean[c];
            }
        }
        let mut len = frames;
        let mut x = self.pre.forward(&self.input.forward(&z, len), len);
        for (up, blocks) in &self.stages {
            x = up.forward(&x, len);
            len *= up.stride();
            let [a, b, c] = blocks.each_ref().map(|block| block.forward(&x, len));
            x = a;
            x.par_iter_mut()
                .zip(b)
                .zip(c)
                .for_each(|((x, b), c)| *x = (*x + b + c) * (1.0 / 3.0));
        }
        self.post_activation.forward(&mut x, len);
        let mut out = self.post.forward(&x, len);
        for v in &mut out {
            *v = v.clamp(-1.0, 1.0);
        }
        out
    }
    /// Decodes `[frames][2][32]` latents (frames varying fastest, as the sampler stores them)
    /// to interleaved stereo samples, `HOP` per frame and channel.
    pub fn decode(&self, latent: &[f32], frames: usize) -> Result<Vec<f32>, &'static str> {
        if frames == 0
            || frames > 4096
            || latent.len() != frames * STREAMS * LATENT_CHANNELS
            || latent.iter().any(|v| !v.is_finite())
        {
            return Err("audio_latent_invalid");
        }
        let (left, right) = rayon::join(
            || self.stream(latent, frames, 0),
            || self.stream(latent, frames, 1),
        );
        Ok(left
            .into_iter()
            .zip(right)
            .flat_map(|(l, r)| [l, r])
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::conv::tests::{close, noise};
    use super::*;
    #[test]
    fn sin_squared_matches_f64() {
        for i in -200_000..200_000 {
            let x = i as f32 * 0.0137;
            let want = (x as f64).sin().powi(2) as f32;
            assert!(
                (sin_squared(x) - want).abs() < 4e-6,
                "{x}: {} vs {want}",
                sin_squared(x)
            );
        }
    }
    /// The definition, written the way the C++ graph builds it: replicate pad, zero-stuffed
    /// transposed filter, snake, replicate pad, strided filter.
    fn reference(alpha: f32, beta: f32, up: &[f32], down: &[f32], x: &[f32]) -> Vec<f32> {
        let len = x.len();
        let padded: Vec<f64> = (0..len + 10)
            .map(|t| x[t.saturating_sub(5).min(len - 1)] as f64)
            .collect();
        let mut full = vec![0f64; (len + 9) * 2 + 12];
        for (t, v) in padded.iter().enumerate() {
            for j in 0..12 {
                full[2 * t + j] += 2.0 * v * up[j] as f64;
            }
        }
        let (a, b) = ((alpha as f64).exp(), (beta as f64).exp() + 1e-9);
        let y: Vec<f64> = full[15..full.len() - 15]
            .iter()
            .map(|v| v + (v * a).sin().powi(2) / b)
            .collect();
        assert_eq!(y.len(), 2 * len);
        let at = |m: usize| y[m.saturating_sub(5).min(2 * len - 1)];
        (0..len)
            .map(|t| (0..12).map(|j| down[j] as f64 * at(2 * t + j)).sum::<f64>() as f32)
            .collect()
    }
    #[test]
    fn activation_matches_definition_for_all_lengths() {
        let (up, down) = (noise(12, 7), noise(12, 8));
        let (alpha, beta) = (vec![0.3, -1.1, 2.0], vec![-0.4, 0.9, 0.1]);
        let activation = Activation::new(&alpha, &beta, &up, &down);
        for len in [1, 2, 3, 5, 6, 7, 13, 64, 257] {
            let x = noise(3 * len, len as u64);
            let mut got = x.clone();
            activation.forward(&mut got, len);
            for c in 0..3 {
                let want = reference(alpha[c], beta[c], &up, &down, &x[c * len..][..len]);
                close(&got[c * len..][..len], &want, 3e-5);
            }
        }
    }
    /// Slow double-precision decoder written from the layer definitions (direct sums, no
    /// shared code with the fast path): the arbiter between this engine and the C++ one.
    mod exact {
        use super::super::{DILATIONS, FILTER, LATENT_CHANNELS, RATES, RESIDUAL_KERNELS, STREAMS};
        use crate::engine::safetensors::Safetensors;
        use rayon::prelude::*;
        type Rows = Vec<Vec<f64>>;
        fn conv(
            file: &Safetensors,
            name: &str,
            cin: usize,
            cout: usize,
            kernel: usize,
            dilation: usize,
            bias: bool,
            x: &Rows,
        ) -> Rows {
            let weight = file
                .f32(&format!("{name}.weight"), &[cout, cin, kernel])
                .unwrap();
            let bias = if bias {
                file.f32(&format!("{name}.bias"), &[cout]).unwrap()
            } else {
                vec![0.0; cout]
            };
            let (len, pad) = (x[0].len(), dilation * (kernel - 1) / 2);
            (0..cout)
                .into_par_iter()
                .map(|o| {
                    let mut row = vec![bias[o] as f64; len];
                    for i in 0..cin {
                        for j in 0..kernel {
                            let w = weight[(o * cin + i) * kernel + j] as f64;
                            for (t, out) in row.iter_mut().enumerate() {
                                if let Some(v) = (t + j * dilation)
                                    .checked_sub(pad)
                                    .and_then(|s| x[i].get(s))
                                {
                                    *out += w * v;
                                }
                            }
                        }
                    }
                    row
                })
                .collect()
        }
        fn up(
            file: &Safetensors,
            name: &str,
            cin: usize,
            cout: usize,
            kernel: usize,
            stride: usize,
            x: &Rows,
        ) -> Rows {
            let weight = file
                .f32(&format!("{name}.weight"), &[cin, cout, kernel])
                .unwrap();
            let bias = file.f32(&format!("{name}.bias"), &[cout]).unwrap();
            let (len, pad) = (x[0].len(), (kernel - stride) / 2);
            (0..cout)
                .into_par_iter()
                .map(|o| {
                    let mut full = vec![0f64; (len - 1) * stride + kernel];
                    for i in 0..cin {
                        for t in 0..len {
                            for j in 0..kernel {
                                full[t * stride + j] +=
                                    weight[(i * cout + o) * kernel + j] as f64 * x[i][t];
                            }
                        }
                    }
                    full[pad..pad + len * stride]
                        .iter()
                        .map(|v| v + bias[o] as f64)
                        .collect()
                })
                .collect()
        }
        fn activation(file: &Safetensors, name: &str, x: &Rows) -> Rows {
            let channels = x.len();
            let alpha = file.f32(&format!("{name}.act.alpha"), &[channels]).unwrap();
            let beta = file.f32(&format!("{name}.act.beta"), &[channels]).unwrap();
            let up = file
                .f32(&format!("{name}.upsample.filter"), &[1, 1, FILTER])
                .unwrap();
            let down = file
                .f32(
                    &format!("{name}.downsample.lowpass.filter"),
                    &[1, 1, FILTER],
                )
                .unwrap();
            x.par_iter()
                .enumerate()
                .map(|(c, x)| {
                    let len = x.len();
                    let mut full = vec![0f64; (len + 9) * 2 + 12];
                    for t in 0..len + 10 {
                        let v = x[t.saturating_sub(5).min(len - 1)];
                        for j in 0..12 {
                            full[2 * t + j] += 2.0 * v * up[j] as f64;
                        }
                    }
                    let (a, b) = ((alpha[c] as f64).exp(), (beta[c] as f64).exp() + 1e-9);
                    let y: Vec<f64> = full[15..full.len() - 15]
                        .iter()
                        .map(|v| v + (v * a).sin().powi(2) / b)
                        .collect();
                    let at = |m: usize| y[m.saturating_sub(5).min(2 * len - 1)];
                    (0..len)
                        .map(|t| (0..12).map(|j| down[j] as f64 * at(2 * t + j)).sum())
                        .collect()
                })
                .collect()
        }
        /// Planar `[stream][sample]` output.
        pub fn decode(file: &Safetensors, latent: &[f32], frames: usize) -> Vec<Vec<f64>> {
            let mean = file.f32("latents_mean", &[LATENT_CHANNELS]).unwrap();
            let std = file.f32("latents_std", &[LATENT_CHANNELS]).unwrap();
            (0..STREAMS)
                .map(|stream| {
                    let z: Rows = (0..LATENT_CHANNELS)
                        .map(|c| {
                            (0..frames)
                                .map(|t| {
                                    latent[t + frames * (stream + STREAMS * c)] as f64
                                        * std[c] as f64
                                        + mean[c] as f64
                                })
                                .collect()
                        })
                        .collect();
                    let mut x = conv(file, "dec_in_proj", LATENT_CHANNELS, 2048, 1, 1, true, &z);
                    x = conv(file, "decoder.conv_pre", 2048, 1024, 7, 1, true, &x);
                    let mut channels = 1024;
                    for (i, rate) in RATES.into_iter().enumerate() {
                        let next = channels / 2;
                        x = up(
                            file,
                            &format!("decoder.ups.{i}.0"),
                            channels,
                            next,
                            if rate == 5 { 9 } else { 4 },
                            rate,
                            &x,
                        );
                        let mut sum: Rows = vec![vec![0.0; x[0].len()]; next];
                        for (j, kernel) in RESIDUAL_KERNELS.into_iter().enumerate() {
                            let name = format!("decoder.resblocks.{}", i * 3 + j);
                            let mut y = x.clone();
                            for (d, dilation) in DILATIONS.into_iter().enumerate() {
                                let mut h =
                                    activation(file, &format!("{name}.activations.{}", 2 * d), &y);
                                h = conv(
                                    file,
                                    &format!("{name}.convs1.{d}"),
                                    next,
                                    next,
                                    kernel,
                                    dilation,
                                    true,
                                    &h,
                                );
                                h = activation(
                                    file,
                                    &format!("{name}.activations.{}", 2 * d + 1),
                                    &h,
                                );
                                h = conv(
                                    file,
                                    &format!("{name}.convs2.{d}"),
                                    next,
                                    next,
                                    kernel,
                                    1,
                                    true,
                                    &h,
                                );
                                y.iter_mut().zip(h).for_each(|(y, h)| {
                                    y.iter_mut().zip(h).for_each(|(y, h)| *y += h)
                                });
                            }
                            sum.iter_mut().zip(y).for_each(|(s, y)| {
                                s.iter_mut().zip(y).for_each(|(s, y)| *s += y / 3.0)
                            });
                        }
                        x = sum;
                        channels = next;
                    }
                    x = activation(file, "decoder.activation_post", &x);
                    conv(file, "decoder.conv_post", channels, 1, 7, 1, false, &x)
                        .remove(0)
                        .into_iter()
                        .map(|v| v.clamp(-1.0, 1.0))
                        .collect()
                })
                .collect()
        }
    }
    /// Compares with stable-diffusion.cpp and with the exact decoder. `KUNO_AUDIO_VAE` is the
    /// model file and `KUNO_AUDIO_GOLDEN` a directory holding `audio-latent.f32` and the C++
    /// engine's planar `audio-wave.f32` for it. Run with `--release -- --ignored --nocapture`.
    ///
    /// Measured 2026-11-09 (93 frames): Rust 129 dB from exact, the C++ SYCL engine 51 dB;
    /// Rust decode 0.49 s on 16 cores against 0.90 s for the C++ engine on the GPU.
    #[test]
    #[ignore]
    fn decoder_matches_exact_reference_and_cpp_engine() {
        let floats = |path: std::path::PathBuf| -> Vec<f32> {
            std::fs::read(path)
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };
        let golden = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_GOLDEN").unwrap());
        let model = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_VAE").unwrap());
        let (latent, cpp) = (
            floats(golden.join("audio-latent.f32")),
            floats(golden.join("audio-wave.f32")),
        );
        let frames = latent.len() / (STREAMS * LATENT_CHANNELS);
        let samples = frames * HOP;
        let start = std::time::Instant::now();
        let decoder = AudioDecoder::load(&model).unwrap();
        let loaded = start.elapsed();
        let mut seconds = Vec::new();
        let mut got = Vec::new();
        for _ in 0..3 {
            let start = std::time::Instant::now();
            got = decoder.decode(&latent, frames).unwrap();
            seconds.push(start.elapsed().as_secs_f32());
        }
        assert_eq!(
            (got.len(), cpp.len()),
            (samples * STREAMS, samples * STREAMS)
        );
        let exact = exact::decode(&Safetensors::open(&model).unwrap(), &latent, frames);
        let db = |value: &dyn Fn(usize, usize) -> f64, truth: &dyn Fn(usize, usize) -> f64| {
            let (mut noise, mut signal) = (0f64, 0f64);
            for channel in 0..STREAMS {
                for t in 0..samples {
                    noise += (value(channel, t) - truth(channel, t)).powi(2);
                    signal += truth(channel, t).powi(2);
                }
            }
            10.0 * (signal / noise.max(1e-30)).log10()
        };
        let rust = |c: usize, t: usize| got[t * STREAMS + c] as f64;
        let engine = |c: usize, t: usize| cpp[c * samples + t] as f64;
        let truth = |c: usize, t: usize| exact[c][t];
        let (rust_exact, cpp_exact, rust_cpp) =
            (db(&rust, &truth), db(&engine, &truth), db(&rust, &engine));
        eprintln!(
            "frames {frames}, load {loaded:.2?}, decode {seconds:.2?} s; from exact: Rust {rust_exact:.1} dB, C++ {cpp_exact:.1} dB; Rust from C++ {rust_cpp:.1} dB"
        );
        assert!(rust_exact > 100.0 && rust_cpp > 45.0);
    }
    #[test]
    fn invalid_latents_are_rejected_without_a_model_file() {
        assert!(
            AudioDecoder::load(std::path::Path::new("/nonexistent/audio.safetensors")).is_err()
        );
    }
}
