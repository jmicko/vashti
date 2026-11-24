//! Qwen Image 2.1's VAE on the GPU (`qwen_vae` holds the layout and the CPU reference).
//! Maps stay channel-major f32, `[channels][pixels]` with the pixel count padded to 256. A
//! convolution unfolds its input into half-float rows (`qvcol`, which also applies a block's
//! RMS norm and SiLU, reads an upsampled map for the decoder's up blocks and strides for the
//! encoder's downsampling) and multiplies them with the half-float weights in `gemm16`,
//! output channels in the token slot so the product lands channel-major; `qvbias` adds the
//! bias and the residual. The middle attention is `gemm16` twice around a softmax. One image
//! is one submission.
use super::{
    dit_gpu::{Call, run},
    gpu::{Buffer, Gpu, Kernel, bytes_of, shaders},
    qwen_vae::{
        ENCODER_TEMPORAL, ENCODER_WIDTHS, LATENT, MEAN, OUT, QwenVae, SCALE, STD, TEMPORAL,
        WAN21_LATENT, WAN21_MEAN, WAN21_SCALE, WAN21_STD, WIDTHS,
    },
    safetensors::narrow,
};
use rayon::prelude::*;

/// Pixel rows of one product are padded to the matrix kernel's 256.
const ROWS: usize = 256;
/// Output channels are padded to the matrix kernel's 128-token tiles.
const TOKENS: usize = 128;
/// Half floats of one unfolded chunk (128 MB).
const CHUNK: usize = 64 << 20;
/// Device memory `decode` leaves free beyond its buffers.
const MARGIN: u64 = 256 << 20;

fn padded(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

/// One convolution's place in the weight and vector buffers.
#[derive(Clone, Copy)]
struct Placed {
    inputs: usize,
    outputs: usize,
    kernel: usize,
    /// Unfolded columns, padded to 16.
    columns: usize,
    weight_first: usize,
    bias_first: usize,
}

struct Placer<'a> {
    vae: &'a QwenVae,
    halves: Vec<u16>,
    vectors: Vec<f32>,
}
impl Placer<'_> {
    fn conv(&mut self, name: &str) -> Result<Placed, &'static str> {
        let file = &self.vae.file;
        let shape = file
            .shape(&format!("{name}.weight"))
            .ok_or("model_tensor_missing")?
            .to_vec();
        let (outputs, inputs, kernel) = (shape[0], shape[1], *shape.last().unwrap_or(&1));
        let mut weight = file.floats(&format!("{name}.weight"), &shape)?;
        let taps = kernel * kernel;
        // A causal 3-D kernel on a single image: only its last time slice sees the image.
        if shape.len() == 5 && shape[2] > 1 {
            let time = shape[2];
            weight = (0..outputs * inputs)
                .flat_map(|oi| {
                    let at = (oi * time + time - 1) * taps;
                    weight[at..at + taps].to_vec()
                })
                .collect();
        }
        let columns = padded(taps * inputs, 16);
        let rows = padded(outputs, TOKENS);
        let weight_first = self.halves.len();
        self.halves.resize(weight_first + rows * columns, 0);
        self.halves[weight_first..]
            .par_chunks_mut(columns)
            .take(outputs)
            .enumerate()
            .for_each(|(o, row)| {
                for i in 0..inputs {
                    for t in 0..taps {
                        row[t * inputs + i] = narrow(weight[(o * inputs + i) * taps + t]);
                    }
                }
            });
        let bias_first = self.vectors.len();
        self.vectors
            .extend(file.floats(&format!("{name}.bias"), &[outputs])?);
        Ok(Placed {
            inputs,
            outputs,
            kernel,
            columns,
            weight_first,
            bias_first,
        })
    }
    fn norm(&mut self, name: &str) -> Result<usize, &'static str> {
        let file = &self.vae.file;
        let shape = file
            .shape(&format!("{name}.gamma"))
            .ok_or("model_tensor_missing")?
            .to_vec();
        let first = self.vectors.len();
        self.vectors
            .extend(file.floats(&format!("{name}.gamma"), &shape)?);
        Ok(first)
    }
    fn residual(&mut self, name: &str) -> Result<Residual, &'static str> {
        Ok(Residual {
            norm1: self.norm(&format!("{name}.residual.0"))?,
            conv1: self.conv(&format!("{name}.residual.2"))?,
            norm2: self.norm(&format!("{name}.residual.3"))?,
            conv2: self.conv(&format!("{name}.residual.6"))?,
            shortcut: self
                .vae
                .file
                .shape(&format!("{name}.shortcut.weight"))
                .map(|_| self.conv(&format!("{name}.shortcut")))
                .transpose()?,
        })
    }
    fn attention(&mut self, name: &str) -> Result<Attention, &'static str> {
        let norm = self.norm(&format!("{name}.norm"))?;
        let to_qkv = self.conv(&format!("{name}.to_qkv"))?;
        let proj = self.conv(&format!("{name}.proj"))?;
        Ok(Attention {
            norm,
            channels: proj.outputs,
            to_qkv,
            proj,
        })
    }
}

struct Residual {
    norm1: usize,
    conv1: Placed,
    norm2: usize,
    conv2: Placed,
    shortcut: Option<Placed>,
}
struct Attention {
    norm: usize,
    channels: usize,
    to_qkv: Placed,
    proj: Placed,
}

/// Up blocks: residual blocks, then the upsampling conv if any.
type Ups = Vec<(Vec<Residual>, Option<Placed>)>;
struct Decoder {
    latent: usize,
    out: usize,
    scale: usize,
    mean: &'static [f32],
    std: &'static [f32],
    /// Qwen Image 2.1's channel-repeating depth-to-space shortcut around each upsampling.
    shortcut: bool,
    conv2: Placed,
    conv1: Placed,
    middle: [Residual; 2],
    attention: Attention,
    /// Five up blocks of three residual blocks, the first four with an upsampling conv.
    ups: Ups,
    head_norm: usize,
    head: Placed,
}
struct Encoder {
    conv1: Placed,
    /// Five down blocks of two residual blocks, the first four with a strided conv.
    downs: Vec<(Vec<Residual>, Option<Placed>)>,
    middle: [Residual; 2],
    attention: Attention,
    head_norm: usize,
    head: Placed,
    /// The VAE's 1x1 conv after the encoder, to means and log-variances.
    quant: Placed,
}

pub struct GpuVae {
    gpu: Gpu,
    col: Kernel,
    rms: Kernel,
    bias: Kernel,
    dup: Kernel,
    avg: Kernel,
    qkv: Kernel,
    soft: Kernel,
    gemm: Kernel,
    weights: Buffer,
    vectors: Buffer,
    decoder: Decoder,
    encoder: Option<Encoder>,
}
impl GpuVae {
    /// Reads the VAE's weights (about 0.35 GB as half floats) into video memory.
    pub fn new(gpu: &Gpu, vae: &QwenVae) -> Result<Self, &'static str> {
        let mut p = Placer {
            vae,
            halves: Vec::new(),
            vectors: Vec::new(),
        };
        let wan21 = vae.wan21;
        let conv2 = p.conv("conv2")?;
        let conv1 = p.conv("decoder.conv1")?;
        let middle = [
            p.residual("decoder.middle.0")?,
            p.residual("decoder.middle.2")?,
        ];
        let attention = p.attention("decoder.middle.1")?;
        // Wan 2.1: four stages of three residual blocks in one list, the first three
        // followed by an upsampling conv.
        let mut wan21_ups = || -> Result<Ups, &'static str> {
            (0..4)
                .map(|stage| {
                    let blocks = (0..3)
                        .map(|i| p.residual(&format!("decoder.upsamples.{}", stage * 4 + i)))
                        .collect::<Result<Vec<_>, _>>()?;
                    let resample = (stage < 3)
                        .then(|| p.conv(&format!("decoder.upsamples.{}.resample.1", stage * 4 + 3)))
                        .transpose()?;
                    Ok((blocks, resample))
                })
                .collect()
        };
        let ups = if wan21 {
            wan21_ups()?
        } else {
            (0..5)
                .map(|block| -> Result<_, &'static str> {
                    let base = format!("decoder.upsamples.{block}.upsamples");
                    let blocks = (0..3)
                        .map(|i| p.residual(&format!("{base}.{i}")))
                        .collect::<Result<Vec<_>, _>>()?;
                    let resample = (block < 4)
                        .then(|| p.conv(&format!("{base}.3.resample.1")))
                        .transpose()?;
                    Ok((blocks, resample))
                })
                .collect::<Result<_, _>>()?
        };
        let decoder = Decoder {
            latent: if wan21 { WAN21_LATENT } else { LATENT },
            out: if wan21 { 3 } else { OUT },
            scale: if wan21 { WAN21_SCALE } else { SCALE },
            mean: if wan21 { &WAN21_MEAN } else { &MEAN },
            std: if wan21 { &WAN21_STD } else { &STD },
            shortcut: !wan21,
            conv2,
            conv1,
            middle,
            attention,
            ups,
            head_norm: p.norm("decoder.head.0")?,
            head: p.conv("decoder.head.2")?,
        };
        let encoder = (!wan21)
            .then(|| -> Result<Encoder, &'static str> {
                Ok(Encoder {
                    conv1: p.conv("encoder.conv1")?,
                    downs: (0..5)
                        .map(|block| -> Result<_, &'static str> {
                            let base = format!("encoder.downsamples.{block}.downsamples");
                            let blocks = (0..2)
                                .map(|i| p.residual(&format!("{base}.{i}")))
                                .collect::<Result<Vec<_>, _>>()?;
                            let resample = (block < 4)
                                .then(|| p.conv(&format!("{base}.2.resample.1")))
                                .transpose()?;
                            Ok((blocks, resample))
                        })
                        .collect::<Result<_, _>>()?,
                    middle: [
                        p.residual("encoder.middle.0")?,
                        p.residual("encoder.middle.2")?,
                    ],
                    attention: p.attention("encoder.middle.1")?,
                    head_norm: p.norm("encoder.head.0")?,
                    head: p.conv("encoder.head.2")?,
                    quant: p.conv("conv1")?,
                })
            })
            .transpose()?;
        let (halves, vectors) = (p.halves, p.vectors);
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
            col: gpu.kernel(shaders::QVCOL, 4, 14, None)?,
            rms: gpu.kernel(shaders::QVRMS, 2, 3, None)?,
            bias: gpu.kernel(shaders::QVBIAS, 4, 4, None)?,
            dup: gpu.kernel(shaders::QVDUP, 2, 7, None)?,
            avg: gpu.kernel(shaders::QVAVG, 2, 7, None)?,
            qkv: gpu.kernel(shaders::QVQKV, 4, 4, None)?,
            soft: gpu.kernel(shaders::QVSOFT, 2, 4, None)?,
            gemm: gpu.gemm16()?,
            weights,
            vectors,
            decoder,
            encoder,
        })
    }

    /// Buffers for maps of at most `largest` values, products of at most `products`, and
    /// attention over `cells` tokens of `channels`.
    fn buffers(
        &self,
        largest: usize,
        products: usize,
        scale_cells: usize,
        cells: usize,
        channels: usize,
    ) -> Result<Buffers, &'static str> {
        let gpu = &self.gpu;
        let bytes = largest as u64 * 4;
        let tokens = padded(cells, ROWS);
        Ok(Buffers {
            maps: [
                gpu.buffer(bytes)?,
                gpu.buffer(bytes)?,
                gpu.buffer(bytes)?,
                gpu.buffer(bytes)?,
            ],
            product: gpu.buffer(products as u64 * 4)?,
            col: gpu.buffer(CHUNK as u64 * 2)?,
            scale: gpu.buffer((padded(scale_cells, ROWS) * 4) as u64)?,
            q: gpu.buffer((tokens * channels * 2) as u64)?,
            k: gpu.buffer((tokens * channels * 2) as u64)?,
            v: gpu.buffer((tokens * channels * 2) as u64)?,
            scores: gpu.buffer((tokens * tokens * 4) as u64)?,
            weights: gpu.buffer((tokens * tokens * 2) as u64)?,
        })
    }

    /// For a latent of `height` x `width`: the decoder's largest map and largest product (in
    /// values: channels times padded pixels at every level, output channels padded to the
    /// matrix kernel's tiles) and the output size.
    fn decode_sizes(&self, height: usize, width: usize) -> (usize, usize, (usize, usize)) {
        let d = &self.decoder;
        let attention = d.attention.channels;
        let level = |channels: usize, w: usize, h: usize| channels * padded(w * h, ROWS);
        let (mut w, mut h) = (width, height);
        let mut largest = level(3 * attention, w, h);
        let mut products = level(padded(3 * attention, TOKENS), w, h);
        for (blocks, resample) in &d.ups {
            let channels = blocks
                .iter()
                .map(|r| r.conv1.inputs.max(r.conv2.outputs))
                .max()
                .unwrap_or(0);
            largest = largest.max(level(channels, w, h));
            products = products.max(level(padded(channels, TOKENS), w, h));
            if let Some(resample) = resample {
                (w, h) = (w * 2, h * 2);
                largest = largest.max(level(resample.outputs, w, h));
                products = products.max(level(padded(resample.outputs, TOKENS), w, h));
            }
        }
        (largest, products, (w, h))
    }
    /// The device memory `decode` takes for a latent of `height` x `width` in one piece
    /// (`buffers`).
    pub fn decode_bytes(&self, height: usize, width: usize) -> u64 {
        let (largest, products, _) = self.decode_sizes(height, width);
        let channels = self.decoder.attention.channels;
        let tokens = padded(height * width, ROWS);
        let (w, h) = (width, height);
        (4 * largest as u64 + products as u64) * 4
            + CHUNK as u64 * 2
            + (padded(w * h * 64, ROWS) * 4) as u64
            + (3 * tokens * channels * 2) as u64
            + (tokens * tokens * 6) as u64
    }
    /// The image `[OUT][16 height][16 width]` in [-1, 1] (before clamping) for a diffusion
    /// latent `[LATENT][height][width]`. When the device's free memory cannot hold the
    /// whole image's buffers, the upsampling path runs in strips of latent rows with enough
    /// rows around each for every convolution's reach (the layers after the attention are
    /// all per pixel or 3 x 3), so the pixels are the same.
    pub fn decode(
        &self,
        latent: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let d = &self.decoder;
        let plane = height * width;
        if latent.len() != d.latent * plane || plane == 0 {
            return Err("invalid_input");
        }
        let need = self.decode_bytes(height, width);
        let budget = self
            .gpu
            .free_memory()
            .map(|free| free.saturating_sub(MARGIN));
        match budget {
            Some(budget) if budget < need => self.decode_strips(latent, height, width, budget),
            _ => self.decode_whole(latent, height, width),
        }
    }

    /// Whether `decode` runs in `free` bytes of device memory without strips of less than
    /// twice their halo (the overlap computed again).
    pub fn fits(&self, height: usize, width: usize, free: u64) -> bool {
        let budget = free.saturating_sub(MARGIN);
        self.decode_bytes(height, width) <= budget
            || self.strip_bytes((4 * self.halo()).min(height), width) <= budget
    }

    /// Latent rows each side of a strip whose outputs every convolution of the upsampling
    /// path still reaches correctly from: each 3 x 3 conv reaches one row at its scale.
    fn halo(&self) -> usize {
        let d = &self.decoder;
        let mut reach = 0f64;
        let mut scale = 1f64;
        for (blocks, resample) in &d.ups {
            for r in blocks {
                reach += (r.conv1.kernel / 2 + r.conv2.kernel / 2) as f64 / scale;
            }
            if let Some(resample) = resample {
                scale *= 2.0;
                reach += (resample.kernel / 2) as f64 / scale;
            }
        }
        reach += (d.head.kernel / 2) as f64 / scale;
        reach.ceil() as usize + 1
    }

    /// `decode` in strips: the latent through the convolutions, middle blocks and attention
    /// whole (at latent size), then each strip of rows with its halo through the rest.
    fn decode_strips(
        &self,
        latent: &[f32],
        height: usize,
        width: usize,
        budget: u64,
    ) -> Result<Vec<f32>, &'static str> {
        let d = &self.decoder;
        let (middle, channels) = self.decode_middle(latent, height, width)?;
        let halo = self.halo();
        // The most rows a strip may own with its halo inside the budget.
        let mut own = height;
        while own > 1 && self.strip_bytes((own + 2 * halo).min(height), width) > budget {
            own = own.div_ceil(2);
        }
        let (scale, out) = (d.scale, d.out);
        let (wide, tall) = (width * scale, height * scale);
        let mut image = vec![0f32; out * wide * tall];
        for first in (0..height).step_by(own) {
            let last = (first + own).min(height);
            let (from, to) = (first.saturating_sub(halo), (last + halo).min(height));
            let rows = to - from;
            let mut slice = vec![0f32; channels * rows * width];
            for c in 0..channels {
                slice[c * rows * width..(c + 1) * rows * width].copy_from_slice(
                    &middle[c * height * width + from * width..c * height * width + to * width],
                );
            }
            let pixels = self.decode_up(&slice, channels, rows, width)?;
            let skip = (first - from) * scale;
            let keep = (last - first) * scale;
            for c in 0..out {
                let strip = &pixels[c * rows * scale * wide..];
                image[c * tall * wide + first * scale * wide..][..keep * wide]
                    .copy_from_slice(&strip[skip * wide..(skip + keep) * wide]);
            }
        }
        Ok(image)
    }

    /// The device memory `decode_up` takes for `rows` latent rows.
    fn strip_bytes(&self, rows: usize, width: usize) -> u64 {
        let (largest, products, _) = self.decode_sizes(rows, width);
        (4 * largest as u64 + products as u64) * 4
            + CHUNK as u64 * 2
            + (padded(rows * width * 64, ROWS) * 4) as u64
    }

    /// Uploads the un-normalized latent, channel-major with padded rows, into `map`.
    fn upload_latent(&self, map: &Buffer, latent: &[f32], plane: usize) -> Result<(), &'static str> {
        let d = &self.decoder;
        let stride = padded(plane, ROWS);
        let mut input = vec![0f32; d.latent * stride];
        for c in 0..d.latent {
            for (to, &v) in input[c * stride..c * stride + plane]
                .iter_mut()
                .zip(&latent[c * plane..(c + 1) * plane])
            {
                *to = v * d.std[c] + d.mean[c];
            }
        }
        self.gpu.upload(map, 0, bytes_of(&input))
    }

    /// The decoder up to its upsampling path at latent size: the map `[channels][plane]`.
    fn decode_middle(
        &self,
        latent: &[f32],
        height: usize,
        width: usize,
    ) -> Result<(Vec<f32>, usize), &'static str> {
        let d = &self.decoder;
        let plane = height * width;
        let attention = d.attention.channels;
        let level = |channels: usize| channels * padded(plane, ROWS);
        let widest = d
            .middle
            .iter()
            .flat_map(|r| [r.conv1.inputs, r.conv1.outputs, r.conv2.outputs])
            .chain([3 * attention, d.conv2.inputs, d.conv2.outputs, d.conv1.outputs])
            .max()
            .unwrap_or(0);
        let b = self.buffers(
            level(widest),
            level(padded(widest, TOKENS)),
            plane,
            plane,
            attention,
        )?;
        self.upload_latent(&b.maps[0], latent, plane)?;
        let work = Work { vae: self, b: &b };
        let mut calls = Vec::new();
        let size = (width, height);
        work.conv(&mut calls, &d.conv2, 0, Same(size), Norm::None, 1, None);
        work.conv(&mut calls, &d.conv1, 1, Same(size), Norm::None, 2, None);
        let mut x = work.residual(&mut calls, &d.middle[0], 2, 2, size);
        x = work.attention(&mut calls, &d.attention, x, size);
        x = work.residual(&mut calls, &d.middle[1], x, x, size);
        run(&self.gpu, &calls)?;
        drop(calls);
        let channels = d.middle[1].conv2.outputs;
        Ok((read_map(&self.gpu, &b.maps[x], channels, plane)?, channels))
    }

    /// The decoder's upsampling path and head on a map `[channels][rows][width]` from
    /// `decode_middle`: the pixels `[OUT][rows * scale][width * scale]`.
    fn decode_up(
        &self,
        middle: &[f32],
        channels: usize,
        rows: usize,
        width: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let gpu = &self.gpu;
        let d = &self.decoder;
        let plane = rows * width;
        let (largest, products, (w, h)) = self.decode_sizes(rows, width);
        let b = self.buffers(largest, products, w * h, 0, d.attention.channels)?;
        let stride = padded(plane, ROWS);
        let mut input = vec![0f32; channels * stride];
        for c in 0..channels {
            input[c * stride..c * stride + plane]
                .copy_from_slice(&middle[c * plane..(c + 1) * plane]);
        }
        gpu.upload(&b.maps[0], 0, bytes_of(&input))?;
        drop(input);
        let work = Work { vae: self, b: &b };
        let mut calls = Vec::new();
        let (w, h, out) = self.up_path(&work, &mut calls, 0, (width, rows));
        run(gpu, &calls)?;
        drop(calls);
        read_map(gpu, &b.maps[out], d.out, w * h)
    }

    /// The upsampling path and head from map `x` of size `size`: the output size and map.
    fn up_path<'a>(
        &'a self,
        work: &Work<'a>,
        calls: &mut Vec<Call<'a>>,
        mut x: usize,
        size: (usize, usize),
    ) -> (usize, usize, usize) {
        let d = &self.decoder;
        let maps = &work.b.maps;
        let (mut w, mut h) = size;
        for (block, (blocks, resample)) in d.ups.iter().enumerate() {
            let input = x;
            for r in blocks {
                x = work.residual(calls, r, x, input, (w, h));
            }
            if let Some(resample) = resample {
                let free = (0..4).find(|i| *i != x && *i != input).unwrap_or_default();
                let (wo, ho) = (w * 2, h * 2);
                work.conv(calls, resample, x, Up((wo, ho)), Norm::None, free, None);
                if d.shortcut {
                    calls.push(Call::Shader {
                        kernel: &self.dup,
                        buffers: vec![&maps[input], &maps[free]],
                        constants: vec![
                            w as u32,
                            h as u32,
                            padded(w * h, ROWS) as u32,
                            padded(wo * ho, ROWS) as u32,
                            WIDTHS[block] as u32,
                            WIDTHS[block + 1] as u32,
                            if TEMPORAL[block] { 2 } else { 1 },
                        ],
                        groups: [(wo * ho).div_ceil(256) as u32, WIDTHS[block + 1] as u32, 1],
                    });
                }
                x = free;
                (w, h) = (wo, ho);
            }
        }
        let out = (0..4).find(|i| *i != x).unwrap_or_default();
        work.conv(calls, &d.head, x, Same((w, h)), Norm::Silu(d.head_norm), out, None);
        (w, h, out)
    }

    /// `decode` in one piece.
    fn decode_whole(
        &self,
        latent: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let gpu = &self.gpu;
        let d = &self.decoder;
        let plane = height * width;
        let attention = d.attention.channels;
        let (largest, products, (w, h)) = self.decode_sizes(height, width);
        let b = self.buffers(largest, products, w * h, plane, attention)?;
        self.upload_latent(&b.maps[0], latent, plane)?;
        let work = Work { vae: self, b: &b };
        let mut calls = Vec::new();
        let size = (width, height);
        work.conv(&mut calls, &d.conv2, 0, Same(size), Norm::None, 1, None);
        work.conv(&mut calls, &d.conv1, 1, Same(size), Norm::None, 2, None);
        let mut x = work.residual(&mut calls, &d.middle[0], 2, 2, size);
        x = work.attention(&mut calls, &d.attention, x, size);
        x = work.residual(&mut calls, &d.middle[1], x, x, size);
        let (w2, h2, out) = self.up_path(&work, &mut calls, x, size);
        run(gpu, &calls)?;
        drop(calls);
        debug_assert_eq!((w2, h2), (w, h));
        debug_assert_eq!((w, h), (width * d.scale, height * d.scale));
        read_map(gpu, &b.maps[out], d.out, w * h)
    }

    /// The diffusion latent `[LATENT][height / 16][width / 16]` of an RGBA image
    /// (interleaved bytes): scaled to [-1, 1], encoded, the means kept and normalized.
    pub fn encode(
        &self,
        rgba: &[u8],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let gpu = &self.gpu;
        let e = self.encoder.as_ref().ok_or("invalid_input")?;
        if height == 0
            || width == 0
            || !height.is_multiple_of(SCALE)
            || !width.is_multiple_of(SCALE)
            || rgba.len() != height * width * OUT
        {
            return Err("invalid_input");
        }
        let plane = height * width;
        let level = |channels: usize, w: usize, h: usize| channels * padded(w * h, ROWS);
        let (mut w, mut h) = (width, height);
        let mut largest = level(ENCODER_WIDTHS[0].max(OUT), w, h);
        let mut products = level(padded(ENCODER_WIDTHS[0], TOKENS), w, h);
        for block in 0..5 {
            let channels = ENCODER_WIDTHS[block].max(ENCODER_WIDTHS[block + 1]);
            largest = largest.max(level(channels, w, h));
            products = products.max(level(padded(channels, TOKENS), w, h));
            if block < 4 {
                (w, h) = (w / 2, h / 2);
            }
        }
        let attention = e.attention.channels;
        largest = largest.max(level(3 * attention, w, h));
        products = products.max(level(padded(3 * attention, TOKENS), w, h));
        let b = self.buffers(largest, products, plane, w * h, attention)?;
        // The image in [-1, 1], channel-major with padded rows.
        let stride = padded(plane, ROWS);
        let mut input = vec![0f32; OUT * stride];
        for (cell, pixel) in rgba.chunks_exact(OUT).enumerate() {
            for (c, byte) in pixel.iter().enumerate() {
                input[c * stride + cell] = f32::from(*byte) / 255.0 * 2.0 - 1.0;
            }
        }
        gpu.upload(&b.maps[0], 0, bytes_of(&input))?;
        drop(input);
        let work = Work { vae: self, b: &b };
        let maps = &b.maps;
        let mut calls = Vec::new();
        let (mut w, mut h) = (width, height);
        work.conv(&mut calls, &e.conv1, 0, Same((w, h)), Norm::None, 1, None);
        let mut x = 1;
        for (block, (blocks, resample)) in e.downs.iter().enumerate() {
            let input = x;
            for r in blocks {
                x = work.residual(&mut calls, r, x, input, (w, h));
            }
            let Some(resample) = resample else {
                // No downsampling: the shortcut is the block's input itself.
                let out = (0..4).find(|i| *i != x && *i != input).unwrap_or_default();
                work.add(&mut calls, x, input, out, ENCODER_WIDTHS[block + 1], (w, h));
                x = out;
                continue;
            };
            let free = (0..4).find(|i| *i != x && *i != input).unwrap_or_default();
            let (wo, ho) = (w / 2, h / 2);
            work.conv(
                &mut calls,
                resample,
                x,
                Down((w, h)),
                Norm::None,
                free,
                None,
            );
            calls.push(Call::Shader {
                kernel: &self.avg,
                buffers: vec![&maps[input], &maps[free]],
                constants: vec![
                    w as u32,
                    h as u32,
                    padded(w * h, ROWS) as u32,
                    padded(wo * ho, ROWS) as u32,
                    ENCODER_WIDTHS[block] as u32,
                    ENCODER_WIDTHS[block + 1] as u32,
                    if ENCODER_TEMPORAL[block] { 2 } else { 1 },
                ],
                groups: [
                    (wo * ho).div_ceil(256) as u32,
                    ENCODER_WIDTHS[block + 1] as u32,
                    1,
                ],
            });
            x = free;
            (w, h) = (wo, ho);
        }
        x = work.residual(&mut calls, &e.middle[0], x, x, (w, h));
        x = work.attention(&mut calls, &e.attention, x, (w, h));
        x = work.residual(&mut calls, &e.middle[1], x, x, (w, h));
        let head = (0..4).find(|i| *i != x).unwrap_or_default();
        work.conv(
            &mut calls,
            &e.head,
            x,
            Same((w, h)),
            Norm::Silu(e.head_norm),
            head,
            None,
        );
        let out = (0..4).find(|i| *i != head).unwrap_or_default();
        work.conv(
            &mut calls,
            &e.quant,
            head,
            Same((w, h)),
            Norm::None,
            out,
            None,
        );
        run(gpu, &calls)?;
        drop(calls);
        let mut latent = read_map(gpu, &maps[out], LATENT, w * h)?;
        let cells = w * h;
        for (c, row) in latent.chunks_exact_mut(cells).enumerate() {
            for v in row {
                *v = (*v - MEAN[c]) / STD[c];
            }
        }
        Ok(latent)
    }
}

/// The first `channels` rows of a channel-major map of `cells` pixels, packed.
fn read_map(
    gpu: &Gpu,
    map: &Buffer,
    channels: usize,
    cells: usize,
) -> Result<Vec<f32>, &'static str> {
    let stride = padded(cells, ROWS);
    let mut values = vec![0f32; channels * stride];
    let through = gpu.readback(values.len() as u64 * 4)?;
    gpu.read_floats(map, 0, &through, &mut values)?;
    let mut packed = vec![0f32; channels * cells];
    for c in 0..channels {
        packed[c * cells..(c + 1) * cells].copy_from_slice(&values[c * stride..c * stride + cells]);
    }
    Ok(packed)
}

/// Scratch of one image.
struct Buffers {
    maps: [Buffer; 4],
    product: Buffer,
    col: Buffer,
    scale: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    scores: Buffer,
    weights: Buffer,
}

/// What a convolution's unfolding applies to its input first.
#[derive(Clone, Copy)]
enum Norm {
    None,
    /// RMS norm over channels with this gamma, then SiLU.
    Silu(usize),
    /// RMS norm only.
    Only(usize),
}
/// A convolution's geometry.
#[derive(Clone, Copy)]
enum Geometry {
    /// Stride 1, padding (k - 1) / 2: the output is the input's size.
    Same((usize, usize)),
    /// Stride 1 over the input upsampled 2x (nearest) to this size.
    Up((usize, usize)),
    /// Stride 2 over an input of this size padded right and bottom by one: half the size.
    Down((usize, usize)),
}
use Geometry::{Down, Same, Up};

struct Work<'a> {
    vae: &'a GpuVae,
    b: &'a Buffers,
}
impl<'a> Work<'a> {
    /// `maps[out] = conv(maps[x]') + bias (+ maps[skip])` where `'` is `norm`.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        calls: &mut Vec<Call<'a>>,
        placed: &'a Placed,
        x: usize,
        geometry: Geometry,
        norm: Norm,
        out: usize,
        skip: Option<usize>,
    ) {
        let e = self.vae;
        let maps = &self.b.maps;
        let ((w_in, h_in), (w_out, h_out), source, stride, pad, up) = match geometry {
            Same((w, h)) => ((w, h), (w, h), w * h, 1, (placed.kernel - 1) / 2, false),
            Up((w, h)) => (
                (w, h),
                (w, h),
                (w / 2) * (h / 2),
                1,
                (placed.kernel - 1) / 2,
                true,
            ),
            Down((w, h)) => ((w, h), (w / 2, h / 2), w * h, 2, 0, false),
        };
        let in_stride = padded(source, ROWS);
        let pixels = w_out * h_out;
        let out_stride = padded(pixels, ROWS);
        let (mode, gamma) = match norm {
            Norm::None => (0, 0),
            Norm::Silu(gamma) => (1, gamma),
            Norm::Only(gamma) => (2, gamma),
        };
        if mode != 0 {
            calls.push(Call::Shader {
                kernel: &e.rms,
                buffers: vec![&maps[x], &self.b.scale],
                constants: vec![placed.inputs as u32, source as u32, in_stride as u32],
                groups: [source.div_ceil(256) as u32, 1, 1],
            });
        }
        let chunk = (CHUNK / placed.columns / ROWS * ROWS).clamp(ROWS, out_stride);
        let rows = padded(placed.outputs, TOKENS);
        for first in (0..out_stride).step_by(chunk) {
            let count = chunk.min(out_stride - first);
            calls.push(Call::Shader {
                kernel: &e.col,
                buffers: vec![&maps[x], &self.b.scale, &e.vectors, &self.b.col],
                constants: vec![
                    placed.inputs as u32,
                    w_in as u32,
                    h_in as u32,
                    in_stride as u32,
                    pixels as u32,
                    first as u32,
                    placed.kernel as u32,
                    placed.columns as u32,
                    mode,
                    gamma as u32,
                    u32::from(up),
                    w_out as u32,
                    stride as u32,
                    pad as u32,
                ],
                groups: [
                    placed.inputs.div_ceil(128) as u32,
                    ROWS as u32,
                    (count / ROWS) as u32,
                ],
            });
            calls.push(Call::Shader {
                kernel: &e.gemm,
                buffers: vec![&e.weights, &self.b.col, &self.b.product],
                constants: vec![
                    placed.columns as u32,
                    placed.weight_first as u32,
                    0,
                    first as u32,
                    out_stride as u32,
                ],
                groups: [(count / ROWS) as u32, (rows / TOKENS) as u32, 1],
            });
        }
        calls.push(Call::Shader {
            kernel: &e.bias,
            buffers: vec![
                &self.b.product,
                &e.vectors,
                &maps[skip.unwrap_or(out)],
                &maps[out],
            ],
            constants: vec![
                pixels as u32,
                out_stride as u32,
                placed.bias_first as u32,
                u32::from(skip.is_some()),
            ],
            groups: [pixels.div_ceil(256) as u32, placed.outputs as u32, 1],
        });
    }
    /// A residual block on map `x`, keeping map `keep` intact; returns the output's map.
    fn residual(
        &self,
        calls: &mut Vec<Call<'a>>,
        r: &'a Residual,
        x: usize,
        keep: usize,
        size: (usize, usize),
    ) -> usize {
        let free: Vec<usize> = (0..4).filter(|i| *i != x && *i != keep).collect();
        let (t, out) = (free[0], free[1]);
        self.conv(calls, &r.conv1, x, Same(size), Norm::Silu(r.norm1), t, None);
        if let Some(shortcut) = &r.shortcut {
            self.conv(calls, shortcut, x, Same(size), Norm::None, out, None);
            self.conv(
                calls,
                &r.conv2,
                t,
                Same(size),
                Norm::Silu(r.norm2),
                out,
                Some(out),
            );
        } else {
            self.conv(
                calls,
                &r.conv2,
                t,
                Same(size),
                Norm::Silu(r.norm2),
                out,
                Some(x),
            );
        }
        out
    }
    /// `maps[out] = maps[a] + maps[b]` over `channels` rows (a block's identity shortcut):
    /// the bias kernel without a bias.
    fn add(
        &self,
        calls: &mut Vec<Call<'a>>,
        a: usize,
        b: usize,
        out: usize,
        channels: usize,
        size: (usize, usize),
    ) {
        let maps = &self.b.maps;
        let pixels = size.0 * size.1;
        calls.push(Call::Shader {
            kernel: &self.vae.bias,
            buffers: vec![&maps[a], &self.vae.vectors, &maps[b], &maps[out]],
            constants: vec![pixels as u32, padded(pixels, ROWS) as u32, u32::MAX, 1],
            groups: [pixels.div_ceil(256) as u32, channels as u32, 1],
        });
    }
    /// Single-head attention over all cells of map `x`, with its residual; returns the
    /// output's map.
    fn attention(
        &self,
        calls: &mut Vec<Call<'a>>,
        a: &'a Attention,
        x: usize,
        size: (usize, usize),
    ) -> usize {
        let e = self.vae;
        let b = self.b;
        let free: Vec<usize> = (0..4).filter(|i| *i != x).collect();
        let (qkv, mixed, out) = (free[0], free[1], free[2]);
        let cells = size.0 * size.1;
        let tokens = padded(cells, ROWS);
        let channels = a.channels;
        self.conv(
            calls,
            &a.to_qkv,
            x,
            Same(size),
            Norm::Only(a.norm),
            qkv,
            None,
        );
        calls.push(Call::Shader {
            kernel: &e.qkv,
            buffers: vec![&b.maps[qkv], &b.q, &b.k, &b.v],
            constants: vec![cells as u32, tokens as u32, tokens as u32, channels as u32],
            groups: [(tokens / 64) as u32, (channels / 4) as u32, 1],
        });
        // Scores [query][key], then the softmax weights.
        calls.push(Call::Shader {
            kernel: &e.gemm,
            buffers: vec![&b.q, &b.k, &b.scores],
            constants: vec![channels as u32, 0, 0, 0, tokens as u32],
            groups: [(tokens / ROWS) as u32, (tokens / TOKENS) as u32, 1],
        });
        calls.push(Call::Shader {
            kernel: &e.soft,
            buffers: vec![&b.scores, &b.weights],
            constants: vec![
                cells as u32,
                tokens as u32,
                0,
                (1.0 / (channels as f32).sqrt()).to_bits(),
            ],
            groups: [tokens as u32, 1, 1],
        });
        // Mixed values, channel-major: [channel][query] = values[channel] . weights[query].
        calls.push(Call::Shader {
            kernel: &e.gemm,
            buffers: vec![&b.v, &b.weights, &b.maps[mixed]],
            constants: vec![tokens as u32, 0, 0, 0, tokens as u32],
            groups: [
                (tokens / ROWS) as u32,
                (padded(channels, TOKENS) / TOKENS) as u32,
                1,
            ],
        });
        self.conv(calls, &a.proj, mixed, Same(size), Norm::None, out, Some(x));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            gpu::Select,
            qwen_dit::tests::{db, read},
            qwen_vae::to_rgba,
        },
        *,
    };

    fn open() -> (QwenVae, GpuVae) {
        let vae = QwenVae::open(std::path::Path::new(
            &std::env::var("KUNO_QWEN_VAE").unwrap(),
        ))
        .unwrap();
        let gpu = Gpu::open(Select::Vendor(std::env::var("KUNO_GPU_VENDOR").map_or(0x8086, |v| u32::from_str_radix(&v, 16).unwrap()))).unwrap();
        let started = std::time::Instant::now();
        let placed = GpuVae::new(&gpu, &vae).unwrap();
        println!("placed in {:.2} s", started.elapsed().as_secs_f64());
        (vae, placed)
    }

    /// `KUNO_QWEN_VAE=<file> [KUNO_QWEN2512_VAE=<dir>] [KUNO_GPU_VENDOR=<hex>]`: decoding in
    /// strips (a budget that fits a few rows) gives the whole decode's pixels, for the Qwen
    /// 2.1 VAE and the Wan 2.1 one.
    #[test]
    #[ignore]
    fn strips_match_whole() {
        let (_, gpu_vae) = open();
        let mut vaes = vec![("2.1", gpu_vae)];
        if let Ok(dir) = std::env::var("KUNO_QWEN2512_VAE") {
            let vae = QwenVae::open_wan21(std::path::Path::new(&dir)).unwrap();
            vaes.push(("2512", GpuVae::new(&vaes[0].1.gpu, &vae).unwrap()));
        }
        for (name, vae) in &vaes {
            let (h, w) = (37, 29);
            let latent: Vec<f32> = (0..vae.decoder.latent * h * w)
                .map(|i| ((i as f32 * 0.618).fract() - 0.5) * 2.0)
                .collect();
            let whole = vae.decode_whole(&latent, h, w).unwrap();
            for strips in [2, 5] {
                let budget = vae.strip_bytes(h.div_ceil(strips) + 2 * vae.halo(), w);
                let parts = vae.decode_strips(&latent, h, w, budget).unwrap();
                let worst = whole
                    .iter()
                    .zip(&parts)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                println!("{name}: about {strips} strips (halo {}), worst difference {worst:e}", vae.halo());
                assert!(worst < 1e-4, "{name}: {worst}");
            }
        }
    }

    /// `KUNO_QWEN_VAE=<model> KUNO_QWEN_VAE_GOLDEN=<dir>`: the GPU decoder against the
    /// engine's decode of the same latent, and its time.
    #[test]
    #[ignore]
    fn gpu_decoder_matches_cpp_engine() {
        let (_, decoder) = open();
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_VAE_GOLDEN").unwrap());
        let shape = std::fs::read_to_string(dir.join("shape.txt")).unwrap();
        let dims: Vec<usize> = shape
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height) = (dims[0], dims[1]);
        let latent = read(&dir.join("latent.f32"));
        let want = read(&dir.join("image.f32"));
        let mut got = Vec::new();
        for run in 0..2 {
            let started = std::time::Instant::now();
            got = decoder.decode(&latent, height, width).unwrap();
            println!("run {run}: {:.2} s", started.elapsed().as_secs_f64());
        }
        let scaled: Vec<f32> = got
            .iter()
            .map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0))
            .collect();
        let db = db(&want, &scaled);
        let pixels = to_rgba(&got, height * SCALE, width * SCALE);
        let want_pixels = to_rgba(
            &want.iter().map(|v| v * 2.0 - 1.0).collect::<Vec<_>>(),
            height * SCALE,
            width * SCALE,
        );
        let diffs: Vec<u8> = pixels
            .iter()
            .zip(&want_pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .collect();
        let worst = diffs.iter().max().unwrap();
        let mean = diffs.iter().map(|d| f64::from(*d)).sum::<f64>() / diffs.len() as f64;
        println!("{width}x{height} latent: {db:.1} dB, worst byte {worst}, mean {mean:.3}");
        assert!(db > 35.0);
    }

    /// `KUNO_QWEN2512_VAE=<qwen_image_vae> KUNO_QWEN2512_GOLDEN=<dir>`: Qwen Image 2512's
    /// (Wan 2.1) decoder against the engine's decode (`vae.latent0`, `vae.pixels0`).
    #[test]
    #[ignore]
    fn wan21_decoder_matches_cpp_engine() {
        let vae = QwenVae::open_wan21(std::path::Path::new(
            &std::env::var("KUNO_QWEN2512_VAE").unwrap(),
        ))
        .unwrap();
        let gpu = Gpu::open(Select::Vendor(std::env::var("KUNO_GPU_VENDOR").map_or(0x8086, |v| u32::from_str_radix(&v, 16).unwrap()))).unwrap();
        let decoder = GpuVae::new(&gpu, &vae).unwrap();
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN2512_GOLDEN").unwrap());
        let dims: Vec<usize> = std::fs::read_to_string(dir.join("vae.latent0.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height) = (dims[0], dims[1]);
        let latent = read(&dir.join("vae.latent0.f32"));
        let want = read(&dir.join("vae.pixels0.f32"));
        let mut got = Vec::new();
        for run in 0..2 {
            let started = std::time::Instant::now();
            got = decoder.decode(&latent, height, width).unwrap();
            println!("run {run}: {:.2} s", started.elapsed().as_secs_f64());
        }
        let scaled: Vec<f32> = got
            .iter()
            .map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0))
            .collect();
        let worst = want
            .iter()
            .zip(&scaled)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let db = db(&want, &scaled);
        println!(
            "{width}x{height} latent: {db:.1} dB, worst {:.0}/255",
            worst * 255.0
        );
        if let Ok(path) = std::env::var("KUNO_QWEN2512_VAE_DUMP") {
            std::fs::write(path, bytes_of(&scaled)).unwrap();
        }
        // The engine decodes in half floats on Vulkan: fine texture differs by a few levels.
        assert!(db > 30.0);
    }

    /// `KUNO_QWEN_VAE=<model> KUNO_QWEN_ENCODE_GOLDEN=<dir>`: the GPU encoder against the
    /// engine's encode of the same image.
    #[test]
    #[ignore]
    fn gpu_encoder_matches_cpp_engine() {
        let (_, placed) = open();
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
        let mut got = Vec::new();
        for run in 0..2 {
            let started = std::time::Instant::now();
            got = placed.encode(&rgba, height, width).unwrap();
            println!("run {run}: {:.3} s", started.elapsed().as_secs_f64());
        }
        println!("{width}x{height}: {:.1} dB", db(&want, &got));
        assert!(db(&want, &got) > 30.0);
    }
}
