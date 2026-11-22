//! MiniMax H3 video encoder for single frames: the start, end and reference images become
//! latents. A stack of causal 3D convolutions with group norms; for one frame the causal
//! padding puts two zero frames before it, so every 3x3x3 kernel reduces to its last time
//! slice, a 3x3 convolution. Spatial padding reflects.
//!
//! This module holds the model's layout, shared with the GPU form (`encoder_gpu`), and the
//! exact form in double precision, which is the arbiter for both the GPU and the C++ engine.
//! The C++ engine encodes in 256-pixel tiles with their own norm statistics; the forms here
//! encode the whole frame, which is the model.
use std::path::Path;

use rayon::prelude::*;

use super::{safetensors::Safetensors, video_vae::LATENT_CHANNELS};

/// Pixels per latent position on each side.
pub const SCALE: usize = 16;
pub(super) const CHANNELS: usize = 128;
pub(super) const LEVELS: usize = 6;
const MULTIPLIERS: [usize; LEVELS] = [1, 2, 2, 4, 4, 8];
/// Levels that end with a stride-2 downsample (the time stride does not matter for a frame).
const DOWNSAMPLES: [bool; LEVELS] = [true, true, true, true, false, false];
pub(super) const GROUPS: usize = 32;
pub(super) const EPS: f64 = 1e-6;
/// The encoder's output: means then log variances of the latent channels.
pub(super) const MOMENTS: usize = LATENT_CHANNELS * 2;
pub(super) const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
pub(super) const STD: [f64; 3] = [0.229, 0.224, 0.225];

/// One convolution: its tensor names' prefix, channels, kernel side (1 or 3), stride and
/// padding (a 3x3 with stride 2 pads only the far sides, by reflection).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Conv {
    pub name: String,
    pub inputs: usize,
    pub outputs: usize,
    pub kernel: usize,
    pub stride: usize,
}
impl Conv {
    fn new(name: String, inputs: usize, outputs: usize, kernel: usize, stride: usize) -> Self {
        Self {
            name,
            inputs,
            outputs,
            kernel,
            stride,
        }
    }
    /// Near-side padding: 1 for a 3x3 at stride 1, none otherwise.
    pub fn pad(&self) -> usize {
        usize::from(self.kernel == 3 && self.stride == 1)
    }
    /// Output side for an input side.
    pub fn out(&self, side: usize) -> usize {
        side / self.stride
    }
}
/// One group norm: its tensor names' prefix and channels.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Norm {
    pub name: String,
    pub channels: usize,
}
/// The encoder's steps in order.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Step {
    /// `x = conv(x)`.
    Conv(Conv),
    /// `x = shortcut(x) + conv2(silu(norm2(conv1(silu(norm1(x))))))`.
    Block {
        norm1: Norm,
        conv1: Conv,
        norm2: Norm,
        conv2: Conv,
        shortcut: Option<Conv>,
    },
    /// `x = conv(silu(norm(x)))`.
    Out { norm: Norm, conv: Conv },
}
pub(super) fn steps() -> Vec<Step> {
    let mut steps = vec![Step::Conv(Conv::new(
        "encoder.conv_in".into(),
        3,
        CHANNELS,
        3,
        1,
    ))];
    let mut previous = CHANNELS;
    for level in 0..LEVELS {
        let current = CHANNELS * MULTIPLIERS[level];
        for block in 0..2 {
            let name = format!("encoder.down.{level}.block.{block}");
            let inputs = if block == 0 { previous } else { current };
            steps.push(Step::Block {
                norm1: Norm {
                    name: format!("{name}.norm1"),
                    channels: inputs,
                },
                conv1: Conv::new(format!("{name}.conv1"), inputs, current, 3, 1),
                norm2: Norm {
                    name: format!("{name}.norm2"),
                    channels: current,
                },
                conv2: Conv::new(format!("{name}.conv2"), current, current, 3, 1),
                shortcut: (inputs != current).then(|| {
                    Conv::new(format!("{name}.nin_shortcut"), inputs, current, 1, 1)
                }),
            });
        }
        if DOWNSAMPLES[level] {
            steps.push(Step::Conv(Conv::new(
                format!("encoder.down.{level}.downsample.conv"),
                current,
                current,
                3,
                2,
            )));
        }
        previous = current;
    }
    steps.push(Step::Out {
        norm: Norm {
            name: "encoder.norm_out".into(),
            channels: previous,
        },
        conv: Conv::new("encoder.conv_out".into(), previous, MOMENTS, 3, 1),
    });
    steps
}

/// The 2D kernel of a causal 3D convolution applied to one frame: `[outputs][inputs][k][k]`,
/// the last time slice of the file's `[outputs][inputs][k][k][k]`.
pub(super) fn kernel_2d(file: &Safetensors, conv: &Conv) -> Result<Vec<f32>, &'static str> {
    let k = conv.kernel;
    let full = file.f16(
        &format!("{}.weight", conv.name),
        &[conv.outputs, conv.inputs, k, k, k],
    )?;
    let slice = k * k;
    Ok(full
        .chunks_exact(k * slice)
        .flat_map(|taps| taps[(k - 1) * slice..].iter().copied())
        .collect())
}
pub(super) fn bias(file: &Safetensors, conv: &Conv) -> Result<Vec<f32>, &'static str> {
    file.f16(&format!("{}.bias", conv.name), &[conv.outputs])
}
/// The 1x1 convolution after the encoder, restricted to the means: weight `[24][48]` and bias.
pub(super) fn quant(file: &Safetensors) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
    let weight = file.f16("quant_conv.weight", &[MOMENTS, MOMENTS, 1, 1, 1])?;
    let bias = file.f16("quant_conv.bias", &[MOMENTS])?;
    Ok((
        weight[..LATENT_CHANNELS * MOMENTS].to_vec(),
        bias[..LATENT_CHANNELS].to_vec(),
    ))
}
/// Reflects an index one step past either edge back inside `0..n`.
pub(super) fn reflect(i: isize, n: usize) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i as usize >= n {
        2 * n - 2 - i as usize
    } else {
        i as usize
    }
}
/// Whether a frame of this size can be encoded: both sides a multiple of 16 and at least 32
/// (the last downsample's reflection needs two rows).
pub fn supported(width: usize, height: usize) -> bool {
    width.is_multiple_of(SCALE) && height.is_multiple_of(SCALE) && width >= 2 * SCALE && height >= 2 * SCALE
}

pub struct VideoEncoder {
    pub(super) file: Safetensors,
}
impl VideoEncoder {
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        file.metadata("minimax_h3_video_vae")
            .ok_or("model_file_invalid")?;
        Ok(Self { file })
    }
    /// Encodes one frame `[3][height][width]` of values in 0..1 to the latent means
    /// `[24][height / 16][width / 16]`, before the diffusion model's normalization. Exact
    /// arithmetic on the file's weights: every value in f64.
    pub fn encode_exact(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f64>, &'static str> {
        if !supported(width, height) || pixels.len() != 3 * width * height {
            return Err("invalid_input");
        }
        let plane = width * height;
        let mut x: Vec<f64> = pixels
            .chunks_exact(plane)
            .enumerate()
            .flat_map(|(c, values)| values.iter().map(move |&v| (v as f64 - MEAN[c]) / STD[c]))
            .collect();
        let (mut w, mut h) = (width, height);
        let f = &self.file;
        for step in steps() {
            match step {
                Step::Conv(conv) => (x, w, h) = exact_conv(f, &conv, &x, w, h)?,
                Step::Block {
                    norm1,
                    conv1,
                    norm2,
                    conv2,
                    shortcut,
                } => {
                    let mut t = x.clone();
                    exact_norm_silu(f, &norm1, &mut t, w * h)?;
                    let (mut t, ..) = exact_conv(f, &conv1, &t, w, h)?;
                    exact_norm_silu(f, &norm2, &mut t, w * h)?;
                    let (t, ..) = exact_conv(f, &conv2, &t, w, h)?;
                    if let Some(shortcut) = shortcut {
                        (x, ..) = exact_conv(f, &shortcut, &x, w, h)?;
                    }
                    x.iter_mut().zip(&t).for_each(|(x, t)| *x += t);
                }
                Step::Out { norm, conv } => {
                    exact_norm_silu(f, &norm, &mut x, w * h)?;
                    (x, ..) = exact_conv(f, &conv, &x, w, h)?;
                }
            }
        }
        let (weight, bias) = quant(f)?;
        let plane = w * h;
        Ok((0..LATENT_CHANNELS)
            .flat_map(|o| {
                let (x, weight, bias) = (&x, &weight, &bias);
                (0..plane).map(move |p| {
                    bias[o] as f64
                        + (0..MOMENTS)
                            .map(|i| weight[o * MOMENTS + i] as f64 * x[i * plane + p])
                            .sum::<f64>()
                })
            })
            .collect())
    }
}
/// `x = silu(group_norm(x))` over channel-major `[channels][plane]`, in f64.
fn exact_norm_silu(
    file: &Safetensors,
    norm: &Norm,
    x: &mut [f64],
    plane: usize,
) -> Result<(), &'static str> {
    let gamma = file.f16(&format!("{}.weight", norm.name), &[norm.channels])?;
    let beta = file.f16(&format!("{}.bias", norm.name), &[norm.channels])?;
    let per = norm.channels / GROUPS;
    x.par_chunks_mut(per * plane)
        .enumerate()
        .for_each(|(g, group)| {
            let n = group.len() as f64;
            let mean = group.iter().sum::<f64>() / n;
            let var = group.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
            let rstd = 1.0 / (var + EPS).sqrt();
            for (c, channel) in group.chunks_exact_mut(plane).enumerate() {
                let (a, b) = (gamma[g * per + c] as f64, beta[g * per + c] as f64);
                for v in channel {
                    let y = (*v - mean) * rstd * a + b;
                    *v = y / (1.0 + (-y).exp());
                }
            }
        });
    Ok(())
}
/// One convolution over channel-major `[inputs][h][w]` in f64; returns the output and its
/// width and height.
fn exact_conv(
    file: &Safetensors,
    conv: &Conv,
    x: &[f64],
    w: usize,
    h: usize,
) -> Result<(Vec<f64>, usize, usize), &'static str> {
    let weight: Vec<f64> = kernel_2d(file, conv)?.into_iter().map(f64::from).collect();
    let bias = bias(file, conv)?;
    let (k, s, pad) = (conv.kernel, conv.stride, conv.pad() as isize);
    let (ow, oh) = (conv.out(w), conv.out(h));
    let mut out = vec![0f64; conv.outputs * ow * oh];
    // Output rows in parallel; each gathers its unfolded inputs once for all outputs.
    let taps = conv.inputs * k * k;
    let rows: Vec<Vec<f64>> = (0..oh)
        .into_par_iter()
        .map(|oy| {
            let mut col = vec![0f64; ow * taps];
            for ox in 0..ow {
                let col = &mut col[ox * taps..(ox + 1) * taps];
                for i in 0..conv.inputs {
                    for ky in 0..k {
                        let iy = reflect((oy * s + ky) as isize - pad, h);
                        for kx in 0..k {
                            let ix = reflect((ox * s + kx) as isize - pad, w);
                            col[(i * k + ky) * k + kx] = x[(i * h + iy) * w + ix];
                        }
                    }
                }
            }
            let mut row = vec![0f64; conv.outputs * ow];
            for o in 0..conv.outputs {
                let weight = &weight[o * taps..(o + 1) * taps];
                for ox in 0..ow {
                    let col = &col[ox * taps..(ox + 1) * taps];
                    row[o * ow + ox] = bias[o] as f64
                        + weight.iter().zip(col).map(|(a, b)| a * b).sum::<f64>();
                }
            }
            row
        })
        .collect();
    for (oy, row) in rows.iter().enumerate() {
        for o in 0..conv.outputs {
            out[(o * oh + oy) * ow..(o * oh + oy + 1) * ow]
                .copy_from_slice(&row[o * ow..(o + 1) * ow]);
        }
    }
    Ok((out, ow, oh))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn encoder_layout_follows_the_model() {
        let steps = steps();
        let convs: Vec<&Conv> = steps
            .iter()
            .flat_map(|step| match step {
                Step::Conv(conv) => vec![conv],
                Step::Block {
                    conv1,
                    conv2,
                    shortcut,
                    ..
                } => [Some(conv1), Some(conv2), shortcut.as_ref()]
                    .into_iter()
                    .flatten()
                    .collect(),
                Step::Out { conv, .. } => vec![conv],
            })
            .collect();
        // conv_in, 12 blocks of two (three where the width changes: levels 1, 3 and 5),
        // four downsamples, conv_out.
        assert_eq!(convs.len(), 1 + 24 + 3 + 4 + 1);
        assert_eq!(convs.iter().filter(|c| c.stride == 2).count(), 4);
        let last = convs.last().unwrap();
        assert_eq!((last.inputs, last.outputs), (1024, MOMENTS));
        // The downsample at stride 2 pads only the far side: 2 * (n/2 - 1) + 2 = n, which
        // reflects to n - 2.
        let down = convs.iter().find(|c| c.stride == 2).unwrap();
        assert_eq!(down.pad(), 0);
        assert_eq!(reflect(2 * (8 - 1) + 2, 16), 14);
        assert_eq!(reflect(-1, 16), 1);
        assert!(supported(832, 544) && !supported(840, 544) && !supported(16, 64));
    }

    /// A smooth synthetic frame: gradients and a few waves, values in 0..1.
    pub fn frame(width: usize, height: usize, seed: u64) -> Vec<f32> {
        let s = seed as f32 * 0.37;
        (0..3)
            .flat_map(|c| {
                (0..height).flat_map(move |y| {
                    (0..width).map(move |x| {
                        let (u, v) = (x as f32 / width as f32, y as f32 / height as f32);
                        let wave = ((u * 9.0 + c as f32 + s).sin() * (v * 7.0 - s).cos()) * 0.3;
                        (0.5 + wave + 0.2 * (u - v) * (c as f32 - 1.0)).clamp(0.0, 1.0)
                    })
                })
            })
            .collect()
    }
    pub fn snr(reference: &[f64], got: &[f32]) -> f64 {
        let signal: f64 = reference.iter().map(|v| v * v).sum();
        let noise: f64 = reference
            .iter()
            .zip(got)
            .map(|(a, &b)| (a - b as f64) * (a - b as f64))
            .sum();
        10.0 * (signal / noise.max(1e-300)).log10()
    }
}
