//! Qwen Image 2.1's VAE decoder on the GPU (`qwen_vae` holds the layout and the CPU
//! reference). Maps stay channel-major f32, `[channels][pixels]` with the pixel count padded
//! to 256. A convolution unfolds its input into half-float rows (`qvcol`, which also applies
//! a block's RMS norm and SiLU, and reads an upsampled map for the up blocks' convolutions)
//! and multiplies them with the half-float weights in `gemm16`, output channels in the token
//! slot so the product lands channel-major; `qvbias` adds the bias and the residual. The
//! middle attention is `gemm16` twice around a softmax. One image is one submission.
use super::{
    dit_gpu::{Call, run},
    gpu::{Buffer, Gpu, Kernel, bytes_of, shaders},
    qwen_vae::{LATENT, MEAN, OUT, QwenVae, SCALE, STD, TEMPORAL, WIDTHS},
    safetensors::narrow,
};

/// Pixel rows of one product are padded to the matrix kernel's 256.
const ROWS: usize = 256;
/// Output channels are padded to the matrix kernel's 128-token tiles.
const TOKENS: usize = 128;
/// Half floats of one unfolded chunk (128 MB).
const CHUNK: usize = 64 << 20;
const ATTENTION: usize = 1152;

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
        let weight = file.floats(&format!("{name}.weight"), &shape)?;
        let taps = kernel * kernel;
        let columns = padded(taps * inputs, 16);
        let rows = padded(outputs, TOKENS);
        let weight_first = self.halves.len();
        self.halves.resize(weight_first + rows * columns, 0);
        let matrix = &mut self.halves[weight_first..];
        for o in 0..outputs {
            for i in 0..inputs {
                for t in 0..taps {
                    matrix[o * columns + t * inputs + i] =
                        narrow(weight[(o * inputs + i) * taps + t]);
                }
            }
        }
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
}

struct Residual {
    norm1: usize,
    conv1: Placed,
    norm2: usize,
    conv2: Placed,
    shortcut: Option<Placed>,
}
fn residual(p: &mut Placer, name: &str) -> Result<Residual, &'static str> {
    Ok(Residual {
        norm1: p.norm(&format!("{name}.residual.0"))?,
        conv1: p.conv(&format!("{name}.residual.2"))?,
        norm2: p.norm(&format!("{name}.residual.3"))?,
        conv2: p.conv(&format!("{name}.residual.6"))?,
        shortcut: p
            .vae
            .file
            .shape(&format!("{name}.shortcut.weight"))
            .map(|_| p.conv(&format!("{name}.shortcut")))
            .transpose()?,
    })
}

pub struct GpuVae {
    gpu: Gpu,
    col: Kernel,
    rms: Kernel,
    bias: Kernel,
    dup: Kernel,
    qkv: Kernel,
    soft: Kernel,
    gemm: Kernel,
    weights: Buffer,
    vectors: Buffer,
    conv2: Placed,
    conv1: Placed,
    middle: [Residual; 2],
    attention_norm: usize,
    to_qkv: Placed,
    proj: Placed,
    /// Five up blocks of three residual blocks, the first four with an upsampling conv.
    ups: Vec<(Vec<Residual>, Option<Placed>)>,
    head_norm: usize,
    head: Placed,
}
impl GpuVae {
    /// Reads the decoder's weights (about 0.3 GB as half floats) into video memory.
    pub fn new(gpu: &Gpu, vae: &QwenVae) -> Result<Self, &'static str> {
        let mut p = Placer {
            vae,
            halves: Vec::new(),
            vectors: Vec::new(),
        };
        let conv2 = p.conv("conv2")?;
        let conv1 = p.conv("decoder.conv1")?;
        let middle = [
            residual(&mut p, "decoder.middle.0")?,
            residual(&mut p, "decoder.middle.2")?,
        ];
        let attention_norm = p.norm("decoder.middle.1.norm")?;
        let to_qkv = p.conv("decoder.middle.1.to_qkv")?;
        let proj = p.conv("decoder.middle.1.proj")?;
        let mut ups = Vec::new();
        for block in 0..5 {
            let base = format!("decoder.upsamples.{block}.upsamples");
            let blocks = (0..3)
                .map(|i| residual(&mut p, &format!("{base}.{i}")))
                .collect::<Result<Vec<_>, _>>()?;
            let resample = (block < 4)
                .then(|| p.conv(&format!("{base}.3.resample.1")))
                .transpose()?;
            ups.push((blocks, resample));
        }
        let head_norm = p.norm("decoder.head.0")?;
        let head = p.conv("decoder.head.2")?;
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
            col: gpu.kernel(shaders::QVCOL, 4, 11, None)?,
            rms: gpu.kernel(shaders::QVRMS, 2, 3, None)?,
            bias: gpu.kernel(shaders::QVBIAS, 4, 4, None)?,
            dup: gpu.kernel(shaders::QVDUP, 2, 7, None)?,
            qkv: gpu.kernel(shaders::QVQKV, 4, 3, None)?,
            soft: gpu.kernel(shaders::QVSOFT, 2, 4, None)?,
            gemm: gpu.kernel(shaders::GEMM16, 3, 5, Some(16))?,
            weights,
            vectors,
            conv2,
            conv1,
            middle,
            attention_norm,
            to_qkv,
            proj,
            ups,
            head_norm,
            head,
        })
    }

    /// The image `[OUT][16 height][16 width]` in [-1, 1] (before clamping) for a diffusion
    /// latent `[LATENT][height][width]`.
    pub fn decode(
        &self,
        latent: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let gpu = &self.gpu;
        let plane = height * width;
        if latent.len() != LATENT * plane || plane == 0 {
            return Err("invalid_input");
        }
        // The largest map (channels times padded pixels at every level), and the largest
        // product (output channels padded to the matrix kernel's tiles).
        let mut w = width;
        let mut h = height;
        let level = |channels: usize, w: usize, h: usize| channels * padded(w * h, ROWS);
        let mut largest = level(3 * ATTENTION, w, h);
        let mut products = level(padded(3 * ATTENTION, TOKENS), w, h);
        for (block, (_, resample)) in self.ups.iter().enumerate() {
            let channels = WIDTHS[block].max(WIDTHS[block + 1]);
            largest = largest.max(level(channels, w, h));
            products = products.max(level(padded(channels, TOKENS), w, h));
            if resample.is_some() {
                (w, h) = (w * 2, h * 2);
                largest = largest.max(level(WIDTHS[block + 1], w, h));
                products = products.max(level(padded(WIDTHS[block + 1], TOKENS), w, h));
            }
        }
        let bytes = largest as u64 * 4;
        let maps = [
            gpu.buffer(bytes)?,
            gpu.buffer(bytes)?,
            gpu.buffer(bytes)?,
            gpu.buffer(bytes)?,
        ];
        let product = gpu.buffer(products as u64 * 4)?;
        let col = gpu.buffer(CHUNK as u64 * 2)?;
        let scale = gpu.buffer((padded(w * h, ROWS) * 4) as u64)?;
        let tokens = padded(plane, ROWS);
        let q = gpu.buffer((tokens * ATTENTION * 2) as u64)?;
        let k = gpu.buffer((tokens * ATTENTION * 2) as u64)?;
        let v = gpu.buffer((tokens * ATTENTION * 2) as u64)?;
        let scores = gpu.buffer((tokens * tokens * 4) as u64)?;
        let weights = gpu.buffer((tokens * tokens * 2) as u64)?;

        // The un-normalized latent, channel-major with padded rows.
        let stride = padded(plane, ROWS);
        let mut input = vec![0f32; LATENT * stride];
        for c in 0..LATENT {
            for (to, &v) in input[c * stride..c * stride + plane]
                .iter_mut()
                .zip(&latent[c * plane..(c + 1) * plane])
            {
                *to = v * STD[c] + MEAN[c];
            }
        }
        gpu.upload(&maps[0], 0, bytes_of(&input))?;
        drop(input);
        let work = Work {
            vae: self,
            product: &product,
            col: &col,
            scale: &scale,
        };
        let mut calls = Vec::new();
        let (w, h) = (width, height);
        work.conv(
            &mut calls,
            &self.conv2,
            &maps[0],
            (w, h),
            Norm::None,
            false,
            &maps[1],
            None,
        );
        work.conv(
            &mut calls,
            &self.conv1,
            &maps[1],
            (w, h),
            Norm::None,
            false,
            &maps[2],
            None,
        );
        let mut x = 2;
        x = work.residual(&mut calls, &self.middle[0], &maps, x, (w, h));
        // Attention over all cells: queries, keys and values from a normalized 1x1 conv.
        let free = (0..4).filter(|i| *i != x).collect::<Vec<_>>();
        let (qkv, mixed, out) = (free[0], free[1], free[2]);
        work.conv(
            &mut calls,
            &self.to_qkv,
            &maps[x],
            (w, h),
            Norm::Only(self.attention_norm),
            false,
            &maps[qkv],
            None,
        );
        calls.push(Call::Shader {
            kernel: &self.qkv,
            buffers: vec![&maps[qkv], &q, &k, &v],
            constants: vec![plane as u32, tokens as u32, stride as u32],
            groups: [(tokens / 64) as u32, (ATTENTION / 4) as u32, 1],
        });
        // Scores [query][key], then the softmax weights.
        calls.push(Call::Shader {
            kernel: &self.gemm,
            buffers: vec![&q, &k, &scores],
            constants: vec![ATTENTION as u32, 0, 0, 0, tokens as u32],
            groups: [(tokens / ROWS) as u32, (tokens / TOKENS) as u32, 1],
        });
        calls.push(Call::Shader {
            kernel: &self.soft,
            buffers: vec![&scores, &weights],
            constants: vec![
                plane as u32,
                tokens as u32,
                0,
                (1.0 / (ATTENTION as f32).sqrt()).to_bits(),
            ],
            groups: [tokens as u32, 1, 1],
        });
        // Mixed values, channel-major: [channel][query] = values[channel][..] . weights[query][..].
        calls.push(Call::Shader {
            kernel: &self.gemm,
            buffers: vec![&v, &weights, &maps[mixed]],
            constants: vec![tokens as u32, 0, 0, 0, stride as u32],
            groups: [(tokens / ROWS) as u32, (ATTENTION / TOKENS) as u32, 1],
        });
        work.conv(
            &mut calls,
            &self.proj,
            &maps[mixed],
            (w, h),
            Norm::None,
            false,
            &maps[out],
            Some(&maps[x]),
        );
        x = out;
        x = work.residual(&mut calls, &self.middle[1], &maps, x, (w, h));
        let (mut w, mut h) = (width, height);
        for (block, (blocks, resample)) in self.ups.iter().enumerate() {
            let input = x;
            for r in blocks {
                x = work.residual_keeping(&mut calls, r, &maps, x, input, (w, h));
            }
            if let Some(resample) = resample {
                let free = (0..4).find(|i| *i != x && *i != input).unwrap_or_default();
                let (wo, ho) = (w * 2, h * 2);
                work.conv(
                    &mut calls,
                    resample,
                    &maps[x],
                    (wo, ho),
                    Norm::None,
                    true,
                    &maps[free],
                    None,
                );
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
                x = free;
                (w, h) = (wo, ho);
            }
        }
        let out = (0..4).find(|i| *i != x).unwrap_or_default();
        work.conv(
            &mut calls,
            &self.head,
            &maps[x],
            (w, h),
            Norm::Silu(self.head_norm),
            false,
            &maps[out],
            None,
        );
        run(gpu, &calls)?;
        drop(calls);
        let pixels = w * h;
        let stride = padded(pixels, ROWS);
        let mut image = vec![0f32; OUT * stride];
        let through = gpu.readback(image.len() as u64 * 4)?;
        gpu.read_floats(&maps[out], 0, &through, &mut image)?;
        let mut packed = vec![0f32; OUT * pixels];
        for c in 0..OUT {
            packed[c * pixels..(c + 1) * pixels]
                .copy_from_slice(&image[c * stride..c * stride + pixels]);
        }
        debug_assert_eq!((w, h), (width * SCALE, height * SCALE));
        Ok(packed)
    }
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

/// The buffers one image shares between its convolutions.
struct Work<'a> {
    vae: &'a GpuVae,
    product: &'a Buffer,
    col: &'a Buffer,
    scale: &'a Buffer,
}
impl<'a> Work<'a> {
    /// `out = conv(x') + bias (+ skip)` where `x'` is `x` (of the convolution's input size
    /// `size`, or half that with `up`) after `norm`.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        calls: &mut Vec<Call<'a>>,
        placed: &'a Placed,
        x: &'a Buffer,
        size: (usize, usize),
        norm: Norm,
        up: bool,
        out: &'a Buffer,
        skip: Option<&'a Buffer>,
    ) {
        let e = self.vae;
        let (w, h) = size;
        let source = if up { (w / 2) * (h / 2) } else { w * h };
        let in_stride = padded(source, ROWS);
        let pixels = w * h;
        let stride = padded(pixels, ROWS);
        let (mode, gamma) = match norm {
            Norm::None => (0, 0),
            Norm::Silu(gamma) => (1, gamma),
            Norm::Only(gamma) => (2, gamma),
        };
        if mode != 0 {
            calls.push(Call::Shader {
                kernel: &e.rms,
                buffers: vec![x, self.scale],
                constants: vec![placed.inputs as u32, source as u32, in_stride as u32],
                groups: [source.div_ceil(256) as u32, 1, 1],
            });
        }
        let chunk = (CHUNK / placed.columns / ROWS * ROWS).clamp(ROWS, stride);
        let rows = padded(placed.outputs, TOKENS);
        for first in (0..stride).step_by(chunk) {
            let count = chunk.min(stride - first);
            calls.push(Call::Shader {
                kernel: &e.col,
                buffers: vec![x, self.scale, &e.vectors, self.col],
                constants: vec![
                    placed.inputs as u32,
                    w as u32,
                    h as u32,
                    in_stride as u32,
                    pixels as u32,
                    first as u32,
                    placed.kernel as u32,
                    placed.columns as u32,
                    mode,
                    gamma as u32,
                    u32::from(up),
                ],
                groups: [
                    placed.inputs.div_ceil(128) as u32,
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
            buffers: vec![self.product, &e.vectors, skip.unwrap_or(out), out],
            constants: vec![
                pixels as u32,
                stride as u32,
                placed.bias_first as u32,
                u32::from(skip.is_some()),
            ],
            groups: [pixels.div_ceil(256) as u32, placed.outputs as u32, 1],
        });
    }
    /// A residual block on map `x`; returns the map holding its output (never `x`'s own
    /// buffer unless the block has no shortcut and nothing else needs `x`).
    fn residual(
        &self,
        calls: &mut Vec<Call<'a>>,
        r: &'a Residual,
        maps: &'a [Buffer; 4],
        x: usize,
        size: (usize, usize),
    ) -> usize {
        self.residual_keeping(calls, r, maps, x, x, size)
    }
    /// As `residual`, keeping map `keep` intact as well.
    fn residual_keeping(
        &self,
        calls: &mut Vec<Call<'a>>,
        r: &'a Residual,
        maps: &'a [Buffer; 4],
        x: usize,
        keep: usize,
        size: (usize, usize),
    ) -> usize {
        let free: Vec<usize> = (0..4).filter(|i| *i != x && *i != keep).collect();
        let (t, out) = (free[0], free[1]);
        self.conv(
            calls,
            &r.conv1,
            &maps[x],
            size,
            Norm::Silu(r.norm1),
            false,
            &maps[t],
            None,
        );
        if let Some(shortcut) = &r.shortcut {
            self.conv(
                calls,
                shortcut,
                &maps[x],
                size,
                Norm::None,
                false,
                &maps[out],
                None,
            );
            self.conv(
                calls,
                &r.conv2,
                &maps[t],
                size,
                Norm::Silu(r.norm2),
                false,
                &maps[out],
                Some(&maps[out]),
            );
        } else {
            self.conv(
                calls,
                &r.conv2,
                &maps[t],
                size,
                Norm::Silu(r.norm2),
                false,
                &maps[out],
                Some(&maps[x]),
            );
        }
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

    /// `KUNO_QWEN_VAE=<model> KUNO_QWEN_VAE_GOLDEN=<dir>`: the GPU decoder against the
    /// engine's decode of the same latent, and its time.
    #[test]
    #[ignore]
    fn gpu_decoder_matches_cpp_engine() {
        let vae = QwenVae::open(std::path::Path::new(
            &std::env::var("KUNO_QWEN_VAE").unwrap(),
        ))
        .unwrap();
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let decoder = GpuVae::new(&gpu, &vae).unwrap();
        println!("placed in {:.2} s", started.elapsed().as_secs_f64());
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
}
