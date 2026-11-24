//! Qwen3-VL's vision tower on the GPU (`vision` holds the layout and the exact form).
//!
//! The patch stream stays f32, `[patches][1152]` with the patch count padded to 512 (so the
//! merged rows, four patches each, fill whole 128-row tiles). Every linear layer is
//! `gemm16` on half-float rows: `vnorm` writes LayerNorm outputs as half floats, `vbias`
//! adds a bias and then the residual, a GELU or the output. Attention runs head by head:
//! `vheads` splits the query/key/value product into half-float operands with the rotary
//! turn, `gemm16` forms the scores, `vsoft` the softmax weights and `gemm16` again the
//! weighted values, which `vgather` lays out for the projection. One image is one
//! submission.
use super::{
    dit_gpu::{Call, run},
    gpu::{Buffer, Gpu, Kernel, bytes_of, shaders},
    safetensors::narrow,
    vision::{
        self, DEEPSTACK, HEAD, HEADS, HIDDEN, LAYERS, MERGED, OUT, PATCH, PATCH_VALUES, TABLE,
        Vision, WIDTH,
    },
};

/// Patches are padded to whole 128-row tiles of merged rows.
const PATCHES: usize = 512;
/// Query and key channels per head, padded to the product's 16.
const QK: usize = 80;
/// Value channel rows per head, padded to the product's 128-token tiles.
const VT: usize = 128;
/// The rotary turn's pairs per head.
const PAIRS: usize = HEAD / 2;
/// Queries per attention block: the score scratch is `[block][patches]`.
const QUERY_BLOCK: usize = 4096;
/// The most patches an image may have (Qwen3-VL's processor allows 12.8 million pixels,
/// about 50 000 patches).
pub const MAX_PATCHES: usize = 65536;

fn padded(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

/// One linear layer's place: `[rows padded to 256][columns]` half floats and its bias.
#[derive(Clone, Copy)]
struct Linear {
    rows: usize,
    columns: usize,
    weight_first: usize,
    bias_first: usize,
}
#[derive(Clone, Copy)]
struct Norm {
    gamma_first: usize,
    beta_first: usize,
}
struct Block {
    norm1: Norm,
    qkv: Linear,
    proj: Linear,
    norm2: Norm,
    fc1: Linear,
    fc2: Linear,
}
struct Merger {
    norm: Norm,
    fc1: Linear,
    fc2: Linear,
}

/// Where one matrix comes from: a tensor of the file, or values computed here.
enum Source {
    File(String),
    Values(Vec<u16>),
}
/// One matrix's region of the weight buffer: `rows * columns` values, then zero rows up to
/// `padded` values.
struct Job {
    source: Source,
    rows: usize,
    columns: usize,
    first: usize,
    padded: usize,
}
/// Lays the weights out first, so they can be converted straight into upload memory.
struct Placer<'a> {
    vision: &'a Vision,
    jobs: Vec<Job>,
    halves: usize,
    vectors: Vec<f32>,
}
impl Placer<'_> {
    fn vector(&mut self, values: &[f32]) -> usize {
        let first = self.vectors.len();
        self.vectors.extend_from_slice(values);
        // Keep vector offsets multiples of 4 for the norm's vec4 rows.
        self.vectors.resize(padded(self.vectors.len(), 4), 0.0);
        first
    }
    fn place(&mut self, source: Source, rows: usize, columns: usize, bias: &[f32]) -> Linear {
        let weight_first = self.halves;
        let size = padded(rows, 256) * columns;
        self.halves += size;
        self.jobs.push(Job {
            source,
            rows,
            columns,
            first: weight_first,
            padded: size,
        });
        Linear {
            rows,
            columns,
            weight_first,
            bias_first: self.vector(bias),
        }
    }
    fn linear(&mut self, name: &str, rows: usize, columns: usize) -> Result<Linear, &'static str> {
        let bias = self.vision.tensor(&format!("{name}.bias"), &[rows])?;
        Ok(self.place(Source::File(format!("visual.{name}.weight")), rows, columns, &bias))
    }
    fn norm(&mut self, name: &str, width: usize) -> Result<Norm, &'static str> {
        let gamma = self.vision.tensor(&format!("{name}.weight"), &[width])?;
        let beta = self.vision.tensor(&format!("{name}.bias"), &[width])?;
        Ok(Norm {
            gamma_first: self.vector(&gamma),
            beta_first: self.vector(&beta),
        })
    }
    fn merger(&mut self, name: &str, width: usize) -> Result<Merger, &'static str> {
        Ok(Merger {
            norm: self.norm(&format!("{name}.norm"), width)?,
            fc1: self.linear(&format!("{name}.linear_fc1"), MERGED, MERGED)?,
            fc2: self.linear(&format!("{name}.linear_fc2"), OUT, MERGED)?,
        })
    }
    /// Writes every matrix as little-endian half floats into `to`, the whole weight buffer.
    fn fill(self, to: &mut [u8]) -> Result<(), &'static str> {
        for job in self.jobs {
            let region = &mut to[job.first * 2..(job.first + job.padded) * 2];
            let (live, pad) = region.split_at_mut(job.rows * job.columns * 2);
            match &job.source {
                Source::File(name) => {
                    self.vision
                        .file
                        .half_into(name, &[job.rows, job.columns], live)?
                }
                Source::Values(values) => {
                    for (to, v) in live.chunks_exact_mut(2).zip(values) {
                        to.copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
            pad.fill(0);
        }
        Ok(())
    }
}

pub struct GpuVision {
    gpu: Gpu,
    norm: Kernel,
    bias: Kernel,
    heads: Kernel,
    soft: Kernel,
    gather: Kernel,
    gemm: Kernel,
    /// Every linear layer's half-float matrix.
    weights: Buffer,
    /// Biases and norm weights, f32.
    vectors: Buffer,
    patch: Linear,
    blocks: Vec<Block>,
    deepstack: Vec<Merger>,
    merger: Merger,
    /// The learned position table `[48 * 48][1152]`, interpolated per image on the CPU.
    table: Vec<f32>,
}
impl GpuVision {
    /// Reads the tower's weights (about 1.2 GB as half floats) into video memory.
    pub fn new(gpu: &Gpu, vision: &Vision) -> Result<Self, &'static str> {
        let mut p = Placer {
            vision,
            jobs: Vec::new(),
            halves: 0,
            vectors: Vec::new(),
        };
        let patch_bias = vision.tensor("patch_embed.proj.bias", &[WIDTH])?;
        let patch_matrix = vision.patch_matrix()?.into_iter().map(narrow).collect();
        let patch = p.place(Source::Values(patch_matrix), WIDTH, PATCH_VALUES, &patch_bias);
        let mut blocks = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let b = format!("blocks.{layer}");
            blocks.push(Block {
                norm1: p.norm(&format!("{b}.norm1"), WIDTH)?,
                qkv: p.linear(&format!("{b}.attn.qkv"), 3 * WIDTH, WIDTH)?,
                proj: p.linear(&format!("{b}.attn.proj"), WIDTH, WIDTH)?,
                norm2: p.norm(&format!("{b}.norm2"), WIDTH)?,
                fc1: p.linear(&format!("{b}.mlp.linear_fc1"), HIDDEN, WIDTH)?,
                fc2: p.linear(&format!("{b}.mlp.linear_fc2"), WIDTH, HIDDEN)?,
            });
        }
        let deepstack = (0..DEEPSTACK.len())
            .map(|i| p.merger(&format!("deepstack_merger_list.{i}"), MERGED))
            .collect::<Result<Vec<_>, _>>()?;
        let merger = p.merger("merger", WIDTH)?;
        let table = vision.tensor("pos_embed.weight", &[TABLE * TABLE, WIDTH])?;
        let vectors = std::mem::take(&mut p.vectors);
        let weights = gpu.filled(p.halves as u64 * 2, |to| p.fill(to))?;
        let vectors = gpu.filled(vectors.len() as u64 * 4, |to| {
            to.copy_from_slice(bytes_of(&vectors));
            Ok(())
        })?;
        Ok(Self {
            gpu: gpu.clone(),
            norm: gpu.kernel(shaders::VNORM, 3, 4, None)?,
            bias: gpu.kernel(shaders::VBIAS, 4, 6, None)?,
            heads: gpu.kernel(shaders::VHEADS, 6, 4, None)?,
            soft: gpu.kernel(shaders::VSOFT, 2, 3, None)?,
            gather: gpu.kernel(shaders::VGATHER, 2, 2, None)?,
            gemm: gpu.kernel(shaders::GEMM16, 3, 5, Some(16))?,
            weights,
            vectors,
            patch,
            blocks,
            deepstack,
            merger,
            table,
        })
    }
    /// The image's rows for the text encoder `[tokens][5120]`, then the three deep-stack
    /// sets, from an image `[3][height][width]` scaled to -1..1.
    pub fn encode(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<[Vec<f32>; 4], &'static str> {
        let n = (width / PATCH) * (height / PATCH);
        if !vision::supported(width, height) || n > MAX_PATCHES || pixels.len() != 3 * width * height
        {
            return Err("invalid_input");
        }
        let gpu = &self.gpu;
        let tp = padded(n, PATCHES);
        let (m, mp) = (n / 4, tp / 4);
        // Inputs: half-float patch rows, the position embeddings as the initial stream, and
        // the rotary turns.
        let mut patch_rows = vec![0u16; tp * PATCH_VALUES];
        for (to, &v) in patch_rows
            .iter_mut()
            .zip(&vision::patches(pixels, width, height))
        {
            *to = narrow(v);
        }
        let mut stream = vec![0f32; tp * WIDTH];
        for (to, &v) in stream
            .iter_mut()
            .zip(&vision::positions(&self.table, width, height))
        {
            *to = v as f32;
        }
        let turns: Vec<f32> = vision::angles(width, height)
            .iter()
            .flat_map(|&a| {
                let (s, c) = a.sin_cos();
                [c as f32, s as f32]
            })
            .collect();
        let half = |count: usize| gpu.buffer(count as u64 * 2);
        let float = |count: usize| gpu.buffer(count as u64 * 4);
        let x = float(tp * WIDTH)?;
        let h = half(tp * WIDTH.max(PATCH_VALUES))?;
        let a = half(tp * HIDDEN)?;
        let g = float(tp * padded(HIDDEN, 256))?;
        let q = half(HEADS * tp * QK)?;
        let k = half(HEADS * tp * QK)?;
        let vt = half(HEADS * VT * tp)?;
        let block = QUERY_BLOCK.min(tp);
        let scores = float(block * tp)?;
        let weights = half(block * tp)?;
        let o = float(HEADS * VT * tp)?;
        let rotary = float(turns.len().max(1))?;
        let outs = [float(m * OUT)?, float(m * OUT)?, float(m * OUT)?, float(m * OUT)?];
        gpu.upload(&h, 0, bytes_of(&patch_rows))?;
        gpu.upload(&x, 0, bytes_of(&stream))?;
        gpu.upload(&rotary, 0, bytes_of(&turns))?;
        drop((patch_rows, stream, turns));

        let w = Work {
            v: self,
            x: &x,
            g: &g,
        };
        let mut calls = Vec::new();
        let c = &mut calls;
        // Patch embedding onto the position embeddings.
        w.product(c, &h, &self.patch, tp);
        w.bias(c, &self.patch, WIDTH, 0, n, tp, None);
        for (layer, b) in self.blocks.iter().enumerate() {
            w.norm(c, b.norm1, WIDTH, n, tp, &h);
            w.product(c, &h, &b.qkv, tp);
            c.push(Call::Shader {
                kernel: &self.heads,
                buffers: vec![&g, &self.vectors, &rotary, &q, &k, &vt],
                constants: vec![
                    n as u32,
                    tp as u32,
                    padded(3 * WIDTH, 256) as u32,
                    b.qkv.bias_first as u32,
                ],
                groups: [tp as u32, HEADS as u32, 1],
            });
            for head in 0..HEADS {
                for first in (0..tp).step_by(block) {
                    let rows = block.min(tp - first);
                    c.push(Call::Shader {
                        kernel: &self.gemm,
                        buffers: vec![&q, &k, &scores],
                        constants: vec![
                            QK as u32,
                            ((head * tp + first) * QK) as u32,
                            (head * tp * QK) as u32,
                            0,
                            tp as u32,
                        ],
                        groups: [(tp / 256) as u32, (rows / 128) as u32, 1],
                    });
                    c.push(Call::Shader {
                        kernel: &self.soft,
                        buffers: vec![&scores, &weights],
                        constants: vec![n as u32, tp as u32, first as u32],
                        groups: [rows as u32, 1, 1],
                    });
                    c.push(Call::Shader {
                        kernel: &self.gemm,
                        buffers: vec![&vt, &weights, &o],
                        constants: vec![
                            tp as u32,
                            (head * VT * tp) as u32,
                            0,
                            (head * VT * tp + first) as u32,
                            tp as u32,
                        ],
                        groups: [(rows / 256) as u32, (VT / 128) as u32, 1],
                    });
                }
            }
            c.push(Call::Shader {
                kernel: &self.gather,
                buffers: vec![&o, &h],
                constants: vec![n as u32, tp as u32],
                groups: [WIDTH.div_ceil(128) as u32, tp as u32, 1],
            });
            w.product(c, &h, &b.proj, tp);
            w.bias(c, &b.proj, WIDTH, 0, n, tp, None);
            w.norm(c, b.norm2, WIDTH, n, tp, &h);
            w.product(c, &h, &b.fc1, tp);
            w.bias(c, &b.fc1, HIDDEN, 1, n, tp, Some(&a));
            w.product(c, &a, &b.fc2, tp);
            w.bias(c, &b.fc2, WIDTH, 0, n, tp, None);
            if let Some(i) = DEEPSTACK.iter().position(|&l| l == layer) {
                // The stream read as merged rows, normed as a whole.
                let d = &self.deepstack[i];
                w.norm(c, d.norm, MERGED, m, mp, &h);
                w.merge(c, &h, &a, d, m, mp, &outs[i + 1]);
            }
        }
        // The final merger norms each patch, then reads four as one row.
        w.norm(c, self.merger.norm, WIDTH, n, tp, &h);
        w.merge(c, &h, &a, &self.merger, m, mp, &outs[0]);
        run(gpu, &calls)?;
        drop(calls);
        let through = gpu.readback((m * OUT) as u64 * 4)?;
        let mut result: [Vec<f32>; 4] = Default::default();
        for (to, from) in result.iter_mut().zip(&outs) {
            *to = vec![0f32; m * OUT];
            gpu.read_floats(from, 0, &through, to)?;
        }
        Ok(result)
    }
}
/// The buffers one image shares between its layers.
struct Work<'a> {
    v: &'a GpuVision,
    /// The f32 patch stream.
    x: &'a Buffer,
    /// Products, f32.
    g: &'a Buffer,
}
impl<'a> Work<'a> {
    /// `g[t][r] = in[t] . weight[r]` over `rows` (padded) tokens.
    fn product(&self, calls: &mut Vec<Call<'a>>, input: &'a Buffer, l: &Linear, rows: usize) {
        let out_rows = padded(l.rows, 256);
        calls.push(Call::Shader {
            kernel: &self.v.gemm,
            buffers: vec![input, &self.v.weights, self.g],
            constants: vec![
                l.columns as u32,
                0,
                l.weight_first as u32,
                0,
                out_rows as u32,
            ],
            groups: [(out_rows / 256) as u32, (rows / 128) as u32, 1],
        });
    }
    /// The product's bias, then by `mode` the residual into the stream (0), a tanh GELU (1)
    /// or an erf GELU (2) into `half`, or the f32 output into `half`'s place (3).
    #[allow(clippy::too_many_arguments)]
    fn bias(
        &self,
        calls: &mut Vec<Call<'a>>,
        l: &Linear,
        width: usize,
        mode: u32,
        count: usize,
        rows: usize,
        out: Option<&'a Buffer>,
    ) {
        let stream = if mode == 3 { out.unwrap_or(self.x) } else { self.x };
        let half = if mode == 1 || mode == 2 {
            out.unwrap_or(self.g)
        } else {
            // Unused by these modes; any half-float-compatible buffer binds.
            self.g
        };
        calls.push(Call::Shader {
            kernel: &self.v.bias,
            buffers: vec![self.g, &self.v.vectors, stream, half],
            constants: vec![
                width as u32,
                padded(l.rows, 256) as u32,
                width as u32,
                l.bias_first as u32,
                count as u32,
                mode,
            ],
            groups: [width.div_ceil(256) as u32, rows as u32, 1],
        });
    }
    /// LayerNorm of the stream's rows of `width` into half floats.
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        calls: &mut Vec<Call<'a>>,
        norm: Norm,
        width: usize,
        count: usize,
        rows: usize,
        out: &'a Buffer,
    ) {
        calls.push(Call::Shader {
            kernel: &self.v.norm,
            buffers: vec![self.x, &self.v.vectors, out],
            constants: vec![
                width as u32,
                count as u32,
                norm.gamma_first as u32,
                norm.beta_first as u32,
            ],
            groups: [rows as u32, 1, 1],
        });
    }
    /// A merger's two linear layers over merged half-float rows `h`, into `out` `[m][5120]`.
    #[allow(clippy::too_many_arguments)]
    fn merge(
        &self,
        calls: &mut Vec<Call<'a>>,
        h: &'a Buffer,
        a: &'a Buffer,
        merger: &'a Merger,
        m: usize,
        mp: usize,
        out: &'a Buffer,
    ) {
        self.product(calls, h, &merger.fc1, mp);
        self.bias(calls, &merger.fc1, MERGED, 2, m, mp, Some(a));
        self.product(calls, a, &merger.fc2, mp);
        self.bias(calls, &merger.fc2, OUT, 3, m, mp, Some(out));
    }
}
const _: () = assert!(
    WIDTH.is_multiple_of(16)
        && HIDDEN.is_multiple_of(16)
        && MERGED.is_multiple_of(256)
        && OUT.is_multiple_of(256)
        && PATCH_VALUES.is_multiple_of(16)
        && PAIRS == 36
        && MAX_PATCHES.is_multiple_of(PATCHES)
        && QUERY_BLOCK.is_multiple_of(PATCHES)
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{gpu::Select, video_encoder::tests::snr, vision::tests::image};

    fn model() -> Option<Vision> {
        let path = std::env::var("KUNO_TEXT_ENCODER").ok()?;
        Some(Vision::open(std::path::Path::new(&path)).unwrap())
    }

    /// The GPU form against the exact one on small images, and its time on a full frame.
    #[test]
    #[ignore]
    fn gpu_vision_matches_exact() {
        let Some(vision) = model() else { return };
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let placed = GpuVision::new(&gpu, &vision).unwrap();
        println!("placed in {:.2} s", started.elapsed().as_secs_f64());
        // The last image has 4352 patches: two attention blocks.
        for (w, h, seed) in [(64, 64, 1), (128, 96, 2), (1088, 1024, 4)] {
            let pixels = image(w, h, seed);
            let exact = vision.encode_exact(&pixels, w, h).unwrap();
            let got = placed.encode(&pixels, w, h).unwrap();
            for (i, (e, g)) in exact.iter().zip(&got).enumerate() {
                println!("{w}x{h} output {i}: {:.1} dB from exact", snr(e, g));
                assert!(snr(e, g) > 30.0);
            }
        }
        let pixels = image(832, 544, 3);
        for _ in 0..3 {
            let started = std::time::Instant::now();
            placed.encode(&pixels, 832, 544).unwrap();
            println!("832x544 in {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        }
    }
}
