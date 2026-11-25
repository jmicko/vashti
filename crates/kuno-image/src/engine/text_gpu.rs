//! Qwen3-VL text encoders on the GPU (Vulkan), with their weights streamed.
//!
//! MiniMax H3's 50 layers hold 24.4 GB of INT8 weights, more than fits beside the DiT, and a prompt
//! is short: computing a layer takes about 6 ms where copying its 488 MB over the bus takes
//! about 44. So the weights stay in host memory (read once) and each layer is copied into a
//! single device window and computed; the device holds one layer, the prompt's activations
//! and every layer's scales and norm weights (16 MB).
//!
//! Reading each layer from the file per prompt instead (into landing areas) was slower:
//! the page-cache reads and the bus copy share the host's memory bandwidth, and a layer
//! took 69 ms where either alone took under 50.
//!
//! The kernels are the DiT's quantizer and integer product (`rotate8`, `gemm8`) and small
//! shaders for the norms, queries and keys, causal attention, residuals and the gate.
use rayon::prelude::*;

use super::{
    dit_gpu::{Call, run},
    gpu::{Buffer, Gpu, Kernel, Staged, bytes_of, shaders},
    text_encoder::{Config, HEAD, PAIRS, Prompt, TextEncoder},
};

/// A model's layer layout.
#[derive(Clone, Copy)]
struct Sizes {
    width: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    inner: usize,
    /// Rows of the layer's products as they sit in the window: queries, keys and values;
    /// the attention output; gate and up; down.
    qkv_rows: usize,
    out_columns: usize,
    gate_rows: usize,
    /// Byte offsets of the four products in the window.
    qkv_at: usize,
    out_at: usize,
    gate_at: usize,
    down_at: usize,
    layer_bytes: usize,
    /// Value offsets in one layer's block of `vectors`: weight scales of the four products,
    /// then the two block norms and the query and key norms.
    scale_qkv: usize,
    scale_out: usize,
    scale_gate: usize,
    scale_down: usize,
    norm1: usize,
    norm2: usize,
    query_norm: usize,
    key_norm: usize,
    /// Query, key and value biases (Qwen2.5-VL).
    bias: usize,
    vectors: usize,
}
impl Sizes {
    fn new(c: Config) -> Self {
        let qkv_rows = (c.heads + 2 * c.kv_heads) * HEAD;
        let out_columns = c.heads * HEAD;
        let gate_rows = 2 * c.inner;
        let out_at = qkv_rows * c.width;
        let gate_at = out_at + c.width * out_columns;
        let down_at = gate_at + gate_rows * c.width;
        let scale_out = qkv_rows;
        let scale_gate = scale_out + c.width;
        let scale_down = scale_gate + gate_rows;
        let norm1 = scale_down + c.width;
        let norm2 = norm1 + c.width;
        let query_norm = norm2 + c.width;
        let key_norm = query_norm + HEAD;
        Self {
            width: c.width,
            layers: c.layers,
            heads: c.heads,
            kv_heads: c.kv_heads,
            inner: c.inner,
            qkv_rows,
            out_columns,
            gate_rows,
            qkv_at: 0,
            out_at,
            gate_at,
            down_at,
            layer_bytes: down_at + c.width * c.inner,
            scale_qkv: 0,
            scale_out,
            scale_gate,
            scale_down,
            norm1,
            norm2,
            query_norm,
            key_norm,
            bias: key_norm + HEAD,
            vectors: key_norm + HEAD + qkv_rows,
        }
    }
}
/// Tokens are padded to the integer product's tile.
const PAD: usize = 128;
/// Weight rows per dispatch of the integer product.
const ROW_BLOCK: usize = 4096;
/// Threads reading the layers from the file.
const READERS: usize = 4;

struct Kernels {
    norm: Kernel,
    rotate8: Kernel,
    gemm8: Kernel,
    qkv: Kernel,
    attend: Kernel,
    resid: Kernel,
    gate: Kernel,
    add: Kernel,
}

/// Buffers of one prompt length.
struct Work {
    tokens: usize,
    padded: usize,
    x: Buffer,
    h: Buffer,
    bytes: Buffer,
    scales: Buffer,
    sums: Buffer,
    query: Buffer,
    key: Buffer,
    value: Buffer,
    turns: Buffer,
    /// Every image's deep-stack rows, `[layer][image rows][width]`.
    deepstack: Buffer,
}

pub struct GpuTextEncoder {
    gpu: Gpu,
    encoder: TextEncoder,
    sizes: Sizes,
    kernels: Kernels,
    /// Every layer's scales and norm weights, `[layers][vectors]`.
    vectors: Buffer,
    window: Buffer,
    /// Every layer's matrices in the window's layout.
    layers: Vec<Staged>,
}
impl GpuTextEncoder {
    pub fn new(gpu: &Gpu, encoder: TextEncoder) -> Result<Self, &'static str> {
        let kernels = Kernels {
            norm: gpu.kernel(shaders::TNORM, 3, 3, gpu.subgroup())?,
            rotate8: gpu.kernel(shaders::ROTATE8, 3, 4, gpu.subgroup())?,
            gemm8: gpu.gemm8()?,
            qkv: gpu.kernel(shaders::TQKV, 8, 9, gpu.subgroup())?,
            attend: if gpu.narrow() {
                gpu.kernel(shaders::TATTEND, 4, 4, Some(16))?
            } else {
                gpu.kernel(shaders::TATTEND32, 4, 4, gpu.subgroup())?
            },
            resid: gpu.kernel(shaders::TRESID, 4, 3, None)?,
            gate: gpu.kernel(shaders::TGATE, 4, 3, None)?,
            add: gpu.kernel(shaders::TADD, 2, 3, None)?,
        };
        let sizes = Sizes::new(encoder.config);
        let readers = rayon::ThreadPoolBuilder::new()
            .num_threads(READERS)
            .build()
            .map_err(|_| "text_failed")?;
        let (layers, scales): (Vec<Staged>, Vec<Option<Vec<f32>>>) = readers
            .install(|| {
                (0..sizes.layers)
                    .into_par_iter()
                    .map(|layer| {
                        let mut scales = None;
                        let staged = gpu.staged(sizes.layer_bytes as u64, |to| {
                            scales = read_layer(&encoder, &sizes, layer, to)?;
                            Ok(())
                        })?;
                        Ok((staged, scales))
                    })
                    .collect::<Result<Vec<_>, &'static str>>()
            })?
            .into_iter()
            .unzip();
        let mut vectors = vec![0f32; sizes.layers * sizes.vectors];
        vectors
            .par_chunks_exact_mut(sizes.vectors)
            .zip(&scales)
            .enumerate()
            .try_for_each(|(layer, (to, scales))| {
                layer_vectors(&encoder, &sizes, layer, to, scales.as_deref())
            })?;
        let vectors = gpu.filled(bytes_of(&vectors).len() as u64, |to| {
            to.copy_from_slice(bytes_of(&vectors));
            Ok(())
        })?;
        Ok(Self {
            window: gpu.buffer(sizes.layer_bytes as u64)?,
            sizes,
            layers,
            vectors,
            kernels,
            encoder,
            gpu: gpu.clone(),
        })
    }
    /// The encoder's sizes.
    pub fn config(&self) -> Config {
        self.encoder.config
    }
    fn work(&self, prompt: &Prompt) -> Result<Work, &'static str> {
        let gpu = &self.gpu;
        let s = &self.sizes;
        let tokens = prompt.ids.len();
        let padded = tokens.next_multiple_of(PAD);
        let f32s = |count: usize| gpu.buffer(count as u64 * 4);
        let rows: usize = prompt.images.iter().map(|i| i.size).sum();
        let layers = prompt
            .images
            .iter()
            .map(|i| i.deepstack.len())
            .max()
            .unwrap_or(0);
        Ok(Work {
            tokens,
            padded,
            x: f32s(padded * s.width)?,
            h: f32s(padded * s.inner.max(s.out_columns))?,
            bytes: gpu.buffer((padded * s.inner.max(s.out_columns)) as u64)?,
            scales: f32s(padded)?,
            sums: f32s(padded * s.gate_rows.max(s.qkv_rows))?,
            query: f32s(s.heads * padded * HEAD)?,
            key: f32s(s.kv_heads * padded * HEAD)?,
            value: f32s(s.kv_heads * padded * HEAD)?,
            turns: f32s(padded * PAIRS * 2)?,
            deepstack: f32s((layers * rows * s.width).max(4))?,
        })
    }
    /// Hidden states after the last layer, `[tokens][width]`.
    pub fn forward(&self, prompt: &Prompt) -> Result<Vec<f32>, &'static str> {
        prompt.check(&self.encoder.config)?;
        let gpu = &self.gpu;
        let width = self.sizes.width;
        let work = self.work(prompt)?;
        let mut x = self.encoder.embed(prompt)?;
        x.resize(work.padded * width, 0.0);
        gpu.upload(&work.x, 0, bytes_of(&x))?;
        x.iter_mut().for_each(|v| *v = 0.0);
        let mut turns = self.encoder.turns(prompt);
        turns.resize(work.padded * PAIRS, [1.0, 0.0]);
        gpu.upload(&work.turns, 0, bytes_of(&turns))?;
        // Deep-stack rows by layer, then image.
        let mut extra = Vec::new();
        let mut adds = Vec::new();
        let layers = prompt
            .images
            .iter()
            .map(|i| i.deepstack.len())
            .max()
            .unwrap_or(0);
        for layer in 0..layers {
            for image in &prompt.images {
                if let Some(rows) = image.deepstack.get(layer) {
                    adds.push((layer, image.index, extra.len() / width, image.size));
                    extra.extend_from_slice(rows);
                }
            }
        }
        if !extra.is_empty() {
            gpu.upload(&work.deepstack, 0, bytes_of(&extra))?;
            extra.iter_mut().for_each(|v| *v = 0.0);
        }

        for layer in 0..self.sizes.layers {
            gpu.copy_staged(&self.layers[layer], &self.window, 0)?;
            let mut calls = self.layer(&work, layer);
            for &(at, index, from, size) in &adds {
                if at == layer {
                    calls.push(Call::Shader {
                        kernel: &self.kernels.add,
                        buffers: vec![&work.x, &work.deepstack],
                        constants: vec![width as u32, index as u32, from as u32],
                        groups: [(width / 4 / 64) as u32, size as u32, 1],
                    });
                }
            }
            run(gpu, &calls)?;
        }
        let mut out = vec![0f32; work.tokens * width];
        let back = gpu.readback((work.tokens * width * 4) as u64)?;
        gpu.read_floats(&work.x, 0, &back, &mut out)?;
        self.encoder.final_norm(&mut out)?;
        Ok(out)
    }
    /// The dispatches of one layer, weights in the window.
    fn layer<'a>(&'a self, w: &'a Work, layer: usize) -> Vec<Call<'a>> {
        let k = &self.kernels;
        let s = &self.sizes;
        let width = s.width;
        let (n, np) = (w.tokens as u32, w.padded as u32);
        let first = layer * s.vectors;
        let norm = |weight: usize| Call::Shader {
            kernel: &k.norm,
            buffers: vec![&w.x, &self.vectors, &w.h],
            constants: vec![n, width as u32, (first + weight) as u32],
            groups: [np, 1, 1],
        };
        let quantize = |width: usize| Call::Shader {
            kernel: &k.rotate8,
            buffers: vec![&w.h, &w.bytes, &w.scales],
            constants: vec![width as u32, (width / 256).div_ceil(16) as u32, 0, 0],
            groups: [np, 1, 1],
        };
        let product = |calls: &mut Vec<Call<'a>>, rows: usize, columns: usize, at: usize| {
            for block in (0..rows).step_by(ROW_BLOCK) {
                calls.push(Call::Shader {
                    kernel: &k.gemm8,
                    buffers: vec![&w.bytes, &self.window, &w.sums],
                    constants: vec![
                        columns as u32,
                        0,
                        (at + block * columns) as u32,
                        block as u32,
                        rows as u32,
                    ],
                    groups: [
                        (ROW_BLOCK.min(rows - block) / 256) as u32,
                        np / PAD as u32,
                        1,
                    ],
                });
            }
        };
        let residual = |scale: usize| Call::Shader {
            kernel: &k.resid,
            buffers: vec![&w.x, &w.sums, &w.scales, &self.vectors],
            constants: vec![n, width as u32, (first + scale) as u32],
            groups: [(width / 4 / 64) as u32, n, 1],
        };
        let mut calls = vec![norm(s.norm1), quantize(width)];
        product(&mut calls, s.qkv_rows, width, s.qkv_at);
        calls.push(Call::Shader {
            kernel: &k.qkv,
            buffers: vec![
                &w.sums,
                &w.scales,
                &self.vectors,
                &self.vectors,
                &w.turns,
                &w.query,
                &w.key,
                &w.value,
            ],
            constants: vec![
                n,
                np,
                (first + s.scale_qkv) as u32,
                (first + s.query_norm) as u32,
                (first + s.key_norm) as u32,
                s.heads as u32,
                s.kv_heads as u32,
                (first + s.bias) as u32,
                if self.encoder.qwen25 { 2 } else { 1 },
            ],
            groups: [(s.heads + 2 * s.kv_heads) as u32, np, 1],
        });
        calls.push(Call::Shader {
            kernel: &k.attend,
            buffers: vec![&w.query, &w.key, &w.value, &w.h],
            constants: vec![n, np, s.heads as u32, (s.heads / s.kv_heads) as u32],
            groups: [np / 4, s.heads as u32, 1],
        });
        calls.push(quantize(s.out_columns));
        product(&mut calls, width, s.out_columns, s.out_at);
        calls.push(residual(s.scale_out));
        calls.push(norm(s.norm2));
        calls.push(quantize(width));
        product(&mut calls, s.gate_rows, width, s.gate_at);
        calls.push(Call::Shader {
            kernel: &k.gate,
            buffers: vec![&w.sums, &w.scales, &self.vectors, &w.h],
            constants: vec![n, (first + s.scale_gate) as u32, (s.inner / 4) as u32],
            groups: [(s.inner / 4).div_ceil(64) as u32, np, 1],
        });
        calls.push(quantize(s.inner));
        product(&mut calls, width, s.inner, s.down_at);
        calls.push(residual(s.scale_down));
        calls
    }
}

/// Reads one layer's matrices into `to` in the window's layout. FP8 matrices (Qwen2.5-VL)
/// are rounded to INT8 in the rotated basis here; their row scales are returned, in the
/// order of the vectors' scales.
fn read_layer(
    encoder: &TextEncoder,
    s: &Sizes,
    layer: usize,
    to: &mut [u8],
) -> Result<Option<Vec<f32>>, &'static str> {
    let base = format!("model.layers.{layer}");
    let parts: [(&str, usize, usize); 7] = [
        ("self_attn.q_proj", s.heads * HEAD, s.width),
        ("self_attn.k_proj", s.kv_heads * HEAD, s.width),
        ("self_attn.v_proj", s.kv_heads * HEAD, s.width),
        ("self_attn.o_proj", s.width, s.out_columns),
        ("mlp.gate_proj", s.inner, s.width),
        ("mlp.up_proj", s.inner, s.width),
        ("mlp.down_proj", s.width, s.inner),
    ];
    let mut rest = &mut to[..s.layer_bytes];
    let mut scales = Vec::new();
    for (name, rows, columns) in parts {
        let (matrix, tail) = rest.split_at_mut(rows * columns);
        if !encoder.is_int8(layer, name) {
            let weight = encoder.float_matrix(layer, name, rows, columns)?;
            let (bytes, row_scales) = crate::engine::dit::quantize(&weight, columns);
            for (to, b) in matrix.iter_mut().zip(&bytes) {
                *to = *b as u8;
            }
            scales.extend(row_scales);
        } else {
            encoder
                .file
                .i8_into(&format!("{base}.{name}.weight"), &[rows, columns], matrix)?;
        }
        rest = tail;
    }
    Ok((!scales.is_empty()).then_some(scales))
}

/// One layer's weight scales and norm weights in the `Sizes::vectors` layout.
fn layer_vectors(
    encoder: &TextEncoder,
    s: &Sizes,
    layer: usize,
    to: &mut [f32],
    scales: Option<&[f32]>,
) -> Result<(), &'static str> {
    let file = &encoder.file;
    let base = format!("model.layers.{layer}");
    let norm = |name: &str, width: usize| file.floats(&format!("{base}.{name}.weight"), &[width]);
    if encoder.qwen25 {
        // Row scales: from rounding at load, else the INT8 file's.
        let scales = match scales {
            Some(scales) => scales.to_vec(),
            None => {
                let mut all = Vec::with_capacity(s.norm1);
                for (name, rows) in [
                    ("self_attn.q_proj", s.heads * HEAD),
                    ("self_attn.k_proj", s.kv_heads * HEAD),
                    ("self_attn.v_proj", s.kv_heads * HEAD),
                    ("self_attn.o_proj", s.width),
                    ("mlp.gate_proj", s.inner),
                    ("mlp.up_proj", s.inner),
                    ("mlp.down_proj", s.width),
                ] {
                    all.extend(file.f32(&format!("{base}.{name}.weight_scale"), &[rows, 1])?);
                }
                all
            }
        };
        to[..scales.len()].copy_from_slice(&scales);
        to[s.norm1..s.norm1 + s.width].copy_from_slice(&norm("input_layernorm", s.width)?);
        to[s.norm2..s.norm2 + s.width].copy_from_slice(&norm("post_attention_layernorm", s.width)?);
        let mut at = s.bias;
        for (name, rows) in [
            ("self_attn.q_proj", s.heads * HEAD),
            ("self_attn.k_proj", s.kv_heads * HEAD),
            ("self_attn.v_proj", s.kv_heads * HEAD),
        ] {
            to[at..at + rows]
                .copy_from_slice(&file.floats(&format!("{base}.{name}.bias"), &[rows])?);
            at += rows;
        }
        return Ok(());
    }
    let scale =
        |name: &str, rows: usize| file.f32(&format!("{base}.{name}.weight_scale"), &[rows, 1]);
    let parts = [
        (s.scale_qkv, scale("self_attn.q_proj", s.heads * HEAD)?),
        (
            s.scale_qkv + s.heads * HEAD,
            scale("self_attn.k_proj", s.kv_heads * HEAD)?,
        ),
        (
            s.scale_qkv + (s.heads + s.kv_heads) * HEAD,
            scale("self_attn.v_proj", s.kv_heads * HEAD)?,
        ),
        (s.scale_out, scale("self_attn.o_proj", s.width)?),
        (s.scale_gate, scale("mlp.gate_proj", s.inner)?),
        (s.scale_gate + s.inner, scale("mlp.up_proj", s.inner)?),
        (s.scale_down, scale("mlp.down_proj", s.width)?),
        (s.norm1, norm("input_layernorm", s.width)?),
        (s.norm2, norm("post_attention_layernorm", s.width)?),
        (s.query_norm, norm("self_attn.q_norm", HEAD)?),
        (s.key_norm, norm("self_attn.k_norm", HEAD)?),
    ];
    for (at, values) in parts {
        to[at..at + values.len()].copy_from_slice(&values);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            gpu::Select,
            text_encoder::tests::{Synthetic, snr},
        },
        *,
    };

    /// Time of each part of a layer: copying it to the device and computing it.
    #[test]
    #[ignore = "needs KUNO_TEXT_ENCODER and the Intel GPU"]
    fn text_layer_costs() {
        let path = std::env::var("KUNO_TEXT_ENCODER").unwrap();
        let open = || TextEncoder::open(std::path::Path::new(&path)).unwrap();
        let reference = open();
        let synthetic = Synthetic::new(&reference, 469, (34, 52));
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let encoder = GpuTextEncoder::new(&gpu, open()).unwrap();
        let prompt = synthetic.prompt();
        let work = encoder.work(&prompt).unwrap();
        for layer in [3, 4, 5] {
            let started = std::time::Instant::now();
            gpu.copy_staged(&encoder.layers[layer], &encoder.window, 0)
                .unwrap();
            let copy = started.elapsed();
            let calls = encoder.layer(&work, layer);
            let started = std::time::Instant::now();
            run(&gpu, &calls).unwrap();
            let compute = started.elapsed();
            println!(
                "layer {layer}: copy {:.1} ms ({:.1} GB/s), compute {:.1} ms",
                copy.as_secs_f64() * 1e3,
                encoder.sizes.layer_bytes as f64 / copy.as_secs_f64() / 1e9,
                compute.as_secs_f64() * 1e3
            );
        }
        // Each kernel of a layer alone.
        let calls = encoder.layer(&work, 3);
        let mut totals = std::collections::BTreeMap::new();
        let k = &encoder.kernels;
        let names = [
            (&k.norm, "norm"),
            (&k.rotate8, "quantize"),
            (&k.gemm8, "products"),
            (&k.qkv, "queries and keys"),
            (&k.attend, "attention"),
            (&k.resid, "residual"),
            (&k.gate, "gate"),
        ];
        for call in &calls {
            let Call::Shader { kernel, .. } = call else {
                continue;
            };
            let name = names.iter().find(|n| std::ptr::eq(n.0, *kernel)).unwrap().1;
            let started = std::time::Instant::now();
            run(&gpu, std::slice::from_ref(call)).unwrap();
            *totals.entry(name).or_insert(0.0) += started.elapsed().as_secs_f64();
        }
        for (name, seconds) in totals {
            println!("{:8.2} ms  {name}", seconds * 1e3);
        }
    }
    /// The GPU form against the CPU reference on a made-up prompt with an image, and its
    /// speed.
    #[test]
    #[ignore = "needs KUNO_TEXT_ENCODER and the Intel GPU"]
    fn gpu_text_matches_reference() {
        let path = std::env::var("KUNO_TEXT_ENCODER").unwrap();
        let open = || TextEncoder::open(std::path::Path::new(&path)).unwrap();
        let reference = open();
        // 469 tokens with a 17 x 26 image, as for a clip at 832 x 544.
        let synthetic = Synthetic::new(&reference, 469, (34, 52));
        let vendor = std::env::var("KUNO_GPU_VENDOR")
            .map_or(0x8086, |v| u32::from_str_radix(&v, 16).unwrap());
        let gpu = Gpu::open(Select::Vendor(vendor)).unwrap();
        println!("{}", gpu.name());
        let started = std::time::Instant::now();
        let encoder = GpuTextEncoder::new(&gpu, open()).unwrap();
        println!("setup {:.2} s", started.elapsed().as_secs_f64());
        let mut got = Vec::new();
        for run in 0..2 {
            let started = std::time::Instant::now();
            got = encoder.forward(&synthetic.prompt()).unwrap();
            println!("run {run}: {:.2} s", started.elapsed().as_secs_f64());
        }
        if std::env::var_os("KUNO_TEXT_SPEED_ONLY").is_some() {
            return;
        }
        let started = std::time::Instant::now();
        let want = reference.forward(&synthetic.prompt()).unwrap();
        println!(
            "CPU reference {:.1} s; GPU {:.1} dB from it",
            started.elapsed().as_secs_f64(),
            snr(&got, &want)
        );
        for token in [0, 5, 200, 449, 468] {
            let width = encoder.sizes.width;
            let row = token * width..(token + 1) * width;
            println!(
                "token {token}: {:.1} dB",
                snr(&got[row.clone()], &want[row])
            );
        }
    }
}
