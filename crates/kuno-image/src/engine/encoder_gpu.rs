//! MiniMax H3 video encoder for single frames on the GPU (`video_encoder` holds the model's
//! layout and the exact form). Activations stay channel-major f32, `[channels][pixels]` with
//! the pixel count padded to 256. A convolution unfolds its input into half-float rows
//! (`ecol`, which also applies the group norm and SiLU before a block's convolutions) and
//! multiplies them with the half-float weights in `gemm16`, output channels in the token
//! slot, so the product lands channel-major; `ebias` adds the bias and the residual. One
//! frame is one submission.
use super::{
    dit_gpu::{Call, run},
    gpu::{Buffer, Gpu, Kernel, bytes_of, shaders},
    safetensors::{Safetensors, narrow},
    video_encoder::{
        self, CHANNELS, Conv, GROUPS, MEAN, MOMENTS, Norm, SCALE, STD, Step, VideoEncoder,
    },
    video_vae::LATENT_CHANNELS,
};

/// Pixel rows of one product are padded to the matrix kernel's 256.
const ROWS: usize = 256;
/// Output channels are padded to the matrix kernel's 128-token tiles.
const TOKENS: usize = 128;
/// Half floats of one unfolded chunk (128 MB).
const CHUNK: usize = 64 << 20;
/// Slices of each group's statistics.
const SLICES: usize = 32;

fn padded(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

/// One convolution's place in the weight and vector buffers.
struct Placed {
    conv: Conv,
    /// Unfolded columns, padded to 16.
    columns: usize,
    weight_first: usize,
    bias_first: usize,
}
struct PlacedNorm {
    channels: usize,
    gamma_first: usize,
    beta_first: usize,
}
enum PlacedStep {
    Conv(Placed),
    Block {
        norm1: PlacedNorm,
        conv1: Placed,
        norm2: PlacedNorm,
        conv2: Placed,
        shortcut: Option<Placed>,
    },
    Out {
        norm: PlacedNorm,
        conv: Placed,
    },
}

/// Appends one convolution's `[outputs padded][columns]` matrix (column `tap * inputs + i`)
/// and its bias.
fn place_conv(
    file: &Safetensors,
    conv: Conv,
    halves: &mut Vec<u16>,
    vectors: &mut Vec<f32>,
) -> Result<Placed, &'static str> {
    let taps = conv.kernel * conv.kernel;
    let columns = padded(taps * conv.inputs, 16);
    let rows = padded(conv.outputs, TOKENS);
    let kernel = video_encoder::kernel_2d(file, &conv)?;
    let weight_first = halves.len();
    halves.resize(weight_first + rows * columns, 0);
    let matrix = &mut halves[weight_first..];
    for o in 0..conv.outputs {
        for i in 0..conv.inputs {
            for t in 0..taps {
                matrix[o * columns + t * conv.inputs + i] =
                    narrow(kernel[(o * conv.inputs + i) * taps + t]);
            }
        }
    }
    let bias_first = vectors.len();
    vectors.extend(video_encoder::bias(file, &conv)?);
    Ok(Placed {
        conv,
        columns,
        weight_first,
        bias_first,
    })
}
fn place_norm(
    file: &Safetensors,
    norm: Norm,
    vectors: &mut Vec<f32>,
) -> Result<PlacedNorm, &'static str> {
    let gamma_first = vectors.len();
    vectors.extend(file.f16(&format!("{}.weight", norm.name), &[norm.channels])?);
    let beta_first = vectors.len();
    vectors.extend(file.f16(&format!("{}.bias", norm.name), &[norm.channels])?);
    Ok(PlacedNorm {
        channels: norm.channels,
        gamma_first,
        beta_first,
    })
}

pub struct GpuEncoder {
    gpu: Gpu,
    col: Kernel,
    stats: Kernel,
    merge: Kernel,
    bias: Kernel,
    gemm: Kernel,
    /// Every convolution's `[outputs padded][columns]` half-float matrix.
    weights: Buffer,
    /// Biases and norm weights, f32.
    vectors: Buffer,
    steps: Vec<PlacedStep>,
    /// The 1x1 convolution to the latent means, run on the CPU: `[24][48]` and bias.
    quant: (Vec<f32>, Vec<f32>),
}
impl GpuEncoder {
    /// Reads the encoder's weights (about 0.4 GB as half floats) into video memory.
    pub fn new(gpu: &Gpu, encoder: &VideoEncoder) -> Result<Self, &'static str> {
        let file = &encoder.file;
        let mut halves: Vec<u16> = Vec::new();
        let mut vectors: Vec<f32> = Vec::new();
        let (h, v) = (&mut halves, &mut vectors);
        let mut placed_steps = Vec::new();
        for step in video_encoder::steps() {
            placed_steps.push(match step {
                Step::Conv(c) => PlacedStep::Conv(place_conv(file, c, h, v)?),
                Step::Block {
                    norm1,
                    conv1,
                    norm2,
                    conv2,
                    shortcut,
                } => PlacedStep::Block {
                    norm1: place_norm(file, norm1, v)?,
                    conv1: place_conv(file, conv1, h, v)?,
                    norm2: place_norm(file, norm2, v)?,
                    conv2: place_conv(file, conv2, h, v)?,
                    shortcut: shortcut.map(|c| place_conv(file, c, h, v)).transpose()?,
                },
                Step::Out { norm, conv } => PlacedStep::Out {
                    norm: place_norm(file, norm, v)?,
                    conv: place_conv(file, conv, h, v)?,
                },
            });
        }
        let weights = gpu.filled(halves.len() as u64 * 2, |to| {
            to.copy_from_slice(bytes_of(&halves));
            Ok(())
        })?;
        let vectors = gpu.filled(vectors.len() as u64 * 4, |to| {
            to.copy_from_slice(bytes_of(&vectors));
            Ok(())
        })?;
        Ok(Self {
            gpu: gpu.clone(),
            col: gpu.kernel(shaders::ECOL, 4, 15, None)?,
            stats: gpu.kernel(shaders::ESTATS, 2, 3, None)?,
            merge: gpu.kernel(shaders::ESTATSEND, 2, 2, None)?,
            bias: gpu.kernel(shaders::EBIAS, 3, 4, None)?,
            gemm: gpu.kernel(shaders::GEMM16, 3, 5, Some(16))?,
            weights,
            vectors,
            steps: placed_steps,
            quant: video_encoder::quant(file)?,
        })
    }
    /// Encodes one frame `[3][height][width]` of values in 0..1 to the latent means
    /// `[24][height / 16][width / 16]`, before the diffusion model's normalization.
    pub fn encode(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, &'static str> {
        if !video_encoder::supported(width, height) || pixels.len() != 3 * width * height {
            return Err("invalid_input");
        }
        let gpu = &self.gpu;
        let plane = width * height;
        let stride = padded(plane, ROWS);
        // The largest activation: channels times padded pixels over every level.
        let mut largest = 3 * stride;
        let (mut w, mut h) = (width, height);
        for step in &self.steps {
            let mut see = |conv: &Conv, w: usize, h: usize| {
                let n = padded(conv.out(w) * conv.out(h), ROWS);
                largest = largest
                    .max(padded(conv.outputs, TOKENS) * n)
                    .max(conv.inputs * padded(w * h, ROWS));
            };
            match step {
                PlacedStep::Conv(c) => {
                    see(&c.conv, w, h);
                    (w, h) = (c.conv.out(w), c.conv.out(h));
                }
                PlacedStep::Block { conv1, conv2, .. } => {
                    see(&conv1.conv, w, h);
                    see(&conv2.conv, w, h);
                }
                PlacedStep::Out { conv, .. } => see(&conv.conv, w, h),
            }
        }
        let bytes = largest as u64 * 4;
        let buffers = [gpu.buffer(bytes)?, gpu.buffer(bytes)?, gpu.buffer(bytes)?];
        let product = gpu.buffer(bytes)?;
        let col = gpu.buffer(CHUNK as u64 * 2)?;
        let partial = gpu.buffer((GROUPS * SLICES * 16) as u64)?;
        let stats = gpu.buffer((GROUPS * 8) as u64)?;
        // The normalized frame, channel-major with padded rows.
        let mut input = vec![0f32; 3 * stride];
        for c in 0..3 {
            for (to, &v) in input[c * stride..c * stride + plane]
                .iter_mut()
                .zip(&pixels[c * plane..(c + 1) * plane])
            {
                *to = ((v as f64 - MEAN[c]) / STD[c]) as f32;
            }
        }
        gpu.upload(&buffers[0], 0, bytes_of(&input))?;
        let work = Work {
            encoder: self,
            product: &product,
            col: &col,
            partial: &partial,
            stats: &stats,
        };
        // x, then two scratch activations.
        let (mut x, mut t, mut u) = (0, 1, 2);
        let (mut w, mut h) = (width, height);
        let mut calls = Vec::new();
        for step in &self.steps {
            match step {
                PlacedStep::Conv(conv) => {
                    work.conv(
                        &mut calls,
                        conv,
                        &buffers[x],
                        w,
                        h,
                        None,
                        &buffers[t],
                        false,
                    );
                    std::mem::swap(&mut x, &mut t);
                    (w, h) = (conv.conv.out(w), conv.conv.out(h));
                }
                PlacedStep::Block {
                    norm1,
                    conv1,
                    norm2,
                    conv2,
                    shortcut,
                } => {
                    let (b, x_, t_, u_) = (&buffers, x, t, u);
                    work.conv(&mut calls, conv1, &b[x_], w, h, Some(norm1), &b[t_], false);
                    if let Some(shortcut) = shortcut {
                        work.conv(&mut calls, shortcut, &b[x_], w, h, None, &b[u_], false);
                        work.conv(&mut calls, conv2, &b[t_], w, h, Some(norm2), &b[u_], true);
                        std::mem::swap(&mut x, &mut u);
                    } else {
                        work.conv(&mut calls, conv2, &b[t_], w, h, Some(norm2), &b[x_], true);
                    }
                }
                PlacedStep::Out { norm, conv } => {
                    work.conv(
                        &mut calls,
                        conv,
                        &buffers[x],
                        w,
                        h,
                        Some(norm),
                        &buffers[t],
                        false,
                    );
                    std::mem::swap(&mut x, &mut t);
                }
            }
        }
        run(gpu, &calls)?;
        drop(calls);
        // The moments, then the means' 1x1 convolution on the CPU.
        let plane = w * h;
        let stride = padded(plane, ROWS);
        let mut moments = vec![0f32; MOMENTS * stride];
        let through = gpu.readback(moments.len() as u64 * 4)?;
        gpu.read_floats(&buffers[x], 0, &through, &mut moments)?;
        drop(through);
        let (weight, bias) = &self.quant;
        let mut latent = vec![0f32; LATENT_CHANNELS * plane];
        for o in 0..LATENT_CHANNELS {
            for p in 0..plane {
                let sum: f64 = (0..MOMENTS)
                    .map(|i| weight[o * MOMENTS + i] as f64 * moments[i * stride + p] as f64)
                    .sum();
                latent[o * plane + p] = (bias[o] as f64 + sum) as f32;
            }
        }
        Ok(latent)
    }
}
/// The buffers one frame shares between its convolutions.
struct Work<'a> {
    encoder: &'a GpuEncoder,
    product: &'a Buffer,
    col: &'a Buffer,
    partial: &'a Buffer,
    stats: &'a Buffer,
}
impl<'a> Work<'a> {
    /// `out = conv(norm ? silu(norm(x)) : x) (+ out)` for `x` of `w * h` pixels.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        calls: &mut Vec<Call<'a>>,
        placed: &'a Placed,
        x: &'a Buffer,
        w: usize,
        h: usize,
        norm: Option<&'a PlacedNorm>,
        out: &'a Buffer,
        add: bool,
    ) {
        let e = self.encoder;
        let conv = &placed.conv;
        let in_stride = padded(w * h, ROWS);
        let (ow, oh) = (conv.out(w), conv.out(h));
        let pixels = ow * oh;
        let stride = padded(pixels, ROWS);
        if let Some(norm) = norm {
            let per = norm.channels / GROUPS;
            calls.push(Call::Shader {
                kernel: &e.stats,
                buffers: vec![x, self.partial],
                constants: vec![per as u32, (w * h) as u32, in_stride as u32],
                groups: [GROUPS as u32, SLICES as u32, 1],
            });
            calls.push(Call::Shader {
                kernel: &e.merge,
                buffers: vec![self.partial, self.stats],
                constants: vec![GROUPS as u32, SLICES as u32],
                groups: [GROUPS.div_ceil(64) as u32, 1, 1],
            });
        }
        let (gamma, beta, per) = norm.map_or((0, 0, 1), |n| {
            (n.gamma_first, n.beta_first, n.channels / GROUPS)
        });
        let chunk = (CHUNK / placed.columns / ROWS * ROWS).clamp(ROWS, stride);
        let rows = padded(conv.outputs, TOKENS);
        for first in (0..stride).step_by(chunk) {
            let count = chunk.min(stride - first);
            calls.push(Call::Shader {
                kernel: &e.col,
                buffers: vec![x, self.stats, &e.vectors, self.col],
                constants: vec![
                    conv.inputs as u32,
                    w as u32,
                    h as u32,
                    in_stride as u32,
                    ow as u32,
                    pixels as u32,
                    first as u32,
                    conv.kernel as u32,
                    conv.stride as u32,
                    conv.pad() as u32,
                    placed.columns as u32,
                    u32::from(norm.is_some()),
                    per as u32,
                    gamma as u32,
                    beta as u32,
                ],
                groups: [
                    conv.inputs.div_ceil(128) as u32,
                    ROWS as u32,
                    (count / ROWS) as u32,
                ],
            });
            calls.push(Call::Shader {
                kernel: &e.gemm,
                buffers: vec![&e.weights, self.col, self.product],
                constants: vec![
                    placed.columns as u32,
                    placed.weight_first as u32,
                    0,
                    first as u32,
                    stride as u32,
                ],
                groups: [(count / ROWS) as u32, (rows / TOKENS) as u32, 1],
            });
        }
        calls.push(Call::Shader {
            kernel: &e.bias,
            buffers: vec![self.product, &e.vectors, out],
            constants: vec![
                pixels as u32,
                stride as u32,
                placed.bias_first as u32,
                u32::from(add),
            ],
            groups: [pixels.div_ceil(256) as u32, conv.outputs as u32, 1],
        });
    }
}
const _: () = assert!(CHANNELS.is_multiple_of(TOKENS) && SCALE == 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{
        gpu::Select,
        video_encoder::tests::{frame, snr},
    };

    fn model() -> Option<VideoEncoder> {
        let path = std::env::var("KUNO_VIDEO_VAE").ok()?;
        Some(VideoEncoder::open(std::path::Path::new(&path)).unwrap())
    }

    /// The GPU form against the exact one on a small frame, and its time on a full one.
    #[test]
    #[ignore]
    fn gpu_encoder_matches_exact() {
        let Some(encoder) = model() else { return };
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let placed = GpuEncoder::new(&gpu, &encoder).unwrap();
        println!("placed in {:.2} s", started.elapsed().as_secs_f64());
        for (w, h, seed) in [(96, 64, 1), (256, 160, 2)] {
            let pixels = frame(w, h, seed);
            let exact = encoder.encode_exact(&pixels, w, h).unwrap();
            let got = placed.encode(&pixels, w, h).unwrap();
            println!("{w}x{h}: {:.1} dB from exact", snr(&exact, &got));
            assert!(snr(&exact, &got) > 30.0);
        }
        let pixels = frame(832, 544, 3);
        for _ in 0..3 {
            let started = std::time::Instant::now();
            placed.encode(&pixels, 832, 544).unwrap();
            println!("832x544 in {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        }
    }
}
