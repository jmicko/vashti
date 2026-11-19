//! MiniMax H3 text encoder on the GPU (Vulkan), with its weights streamed.
//!
//! The 50 layers hold 24.4 GB of INT8 weights, more than fits beside the DiT, and a prompt
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
    text_encoder::{
        HEAD, HEADS, INNER, KV_HEADS, LAYERS, PAIRS, Prompt, TextEncoder, WIDTH, positions, rotary,
    },
};

/// Rows of the layer's products as they sit in the window: queries, keys and values; the
/// attention output; gate and up; down.
const QKV_ROWS: usize = (HEADS + 2 * KV_HEADS) * HEAD;
const OUT_COLUMNS: usize = HEADS * HEAD;
const GATE_ROWS: usize = 2 * INNER;
/// Byte offsets of the four products in the window.
const QKV_AT: usize = 0;
const OUT_AT: usize = QKV_AT + QKV_ROWS * WIDTH;
const GATE_AT: usize = OUT_AT + WIDTH * OUT_COLUMNS;
const DOWN_AT: usize = GATE_AT + GATE_ROWS * WIDTH;
const LAYER_BYTES: usize = DOWN_AT + WIDTH * INNER;
/// Value offsets in one layer's block of `vectors`: weight scales of the four products, then
/// the two block norms and the query and key norms.
const SCALE_QKV: usize = 0;
const SCALE_OUT: usize = SCALE_QKV + QKV_ROWS;
const SCALE_GATE: usize = SCALE_OUT + WIDTH;
const SCALE_DOWN: usize = SCALE_GATE + GATE_ROWS;
const NORM1: usize = SCALE_DOWN + WIDTH;
const NORM2: usize = NORM1 + WIDTH;
const QUERY_NORM: usize = NORM2 + WIDTH;
const KEY_NORM: usize = QUERY_NORM + HEAD;
const VECTORS: usize = KEY_NORM + HEAD;
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
    /// Every image's deep-stack rows, `[layer][image rows][WIDTH]`.
    deepstack: Buffer,
}

pub struct GpuTextEncoder {
    gpu: Gpu,
    encoder: TextEncoder,
    kernels: Kernels,
    /// Every layer's scales and norm weights, `[LAYERS][VECTORS]`.
    vectors: Buffer,
    window: Buffer,
    /// Every layer's matrices in the window's layout.
    layers: Vec<Staged>,
}
impl GpuTextEncoder {
    pub fn new(gpu: &Gpu, encoder: TextEncoder) -> Result<Self, &'static str> {
        let kernels = Kernels {
            norm: gpu.kernel(shaders::TNORM, 3, 3, Some(16))?,
            rotate8: gpu.kernel(shaders::ROTATE8, 3, 3, Some(16))?,
            gemm8: gpu.kernel(shaders::GEMM8, 3, 5, Some(16))?,
            qkv: gpu.kernel(shaders::TQKV, 8, 5, Some(16))?,
            attend: gpu.kernel(shaders::TATTEND, 4, 2, Some(16))?,
            resid: gpu.kernel(shaders::TRESID, 4, 3, None)?,
            gate: gpu.kernel(shaders::TGATE, 4, 2, None)?,
            add: gpu.kernel(shaders::TADD, 2, 3, None)?,
        };
        let mut vectors = vec![0f32; LAYERS * VECTORS];
        vectors
            .par_chunks_exact_mut(VECTORS)
            .enumerate()
            .try_for_each(|(layer, to)| layer_vectors(&encoder, layer, to))?;
        let vectors = gpu.filled(bytes_of(&vectors).len() as u64, |to| {
            to.copy_from_slice(bytes_of(&vectors));
            Ok(())
        })?;
        let readers = rayon::ThreadPoolBuilder::new()
            .num_threads(READERS)
            .build()
            .map_err(|_| "text_failed")?;
        let layers = readers.install(|| {
            (0..LAYERS)
                .into_par_iter()
                .map(|layer| {
                    gpu.staged(LAYER_BYTES as u64, |to| read_layer(&encoder, layer, to))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        Ok(Self {
            window: gpu.buffer(LAYER_BYTES as u64)?,
            layers,
            vectors,
            kernels,
            encoder,
            gpu: gpu.clone(),
        })
    }
    fn work(&self, prompt: &Prompt) -> Result<Work, &'static str> {
        let gpu = &self.gpu;
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
            x: f32s(padded * WIDTH)?,
            h: f32s(padded * INNER)?,
            bytes: gpu.buffer((padded * INNER) as u64)?,
            scales: f32s(padded)?,
            sums: f32s(padded * GATE_ROWS)?,
            query: f32s(HEADS * padded * HEAD)?,
            key: f32s(KV_HEADS * padded * HEAD)?,
            value: f32s(KV_HEADS * padded * HEAD)?,
            turns: f32s(padded * PAIRS * 2)?,
            deepstack: f32s((layers * rows * WIDTH).max(4))?,
        })
    }
    /// Hidden states after the last layer, `[tokens][WIDTH]`.
    pub fn forward(&self, prompt: &Prompt) -> Result<Vec<f32>, &'static str> {
        prompt.check()?;
        let gpu = &self.gpu;
        let work = self.work(prompt)?;
        let mut x = self.encoder.embed(prompt)?;
        x.resize(work.padded * WIDTH, 0.0);
        gpu.upload(&work.x, 0, bytes_of(&x))?;
        x.iter_mut().for_each(|v| *v = 0.0);
        let mut turns = rotary(&positions(prompt));
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
                    adds.push((layer, image.index, extra.len() / WIDTH, image.size));
                    extra.extend_from_slice(rows);
                }
            }
        }
        if !extra.is_empty() {
            gpu.upload(&work.deepstack, 0, bytes_of(&extra))?;
            extra.iter_mut().for_each(|v| *v = 0.0);
        }

        for layer in 0..LAYERS {
            gpu.copy_staged(&self.layers[layer], &self.window, 0)?;
            let mut calls = self.layer(&work, layer);
            for &(at, index, from, size) in &adds {
                if at == layer {
                    calls.push(Call::Shader {
                        kernel: &self.kernels.add,
                        buffers: vec![&work.x, &work.deepstack],
                        constants: vec![WIDTH as u32, index as u32, from as u32],
                        groups: [(WIDTH / 4 / 64) as u32, size as u32, 1],
                    });
                }
            }
            run(gpu, &calls)?;
        }
        let mut out = vec![0f32; work.tokens * WIDTH];
        let back = gpu.readback((work.tokens * WIDTH * 4) as u64)?;
        gpu.read_floats(&work.x, 0, &back, &mut out)?;
        Ok(out)
    }
    /// The dispatches of one layer, weights in the window.
    fn layer<'a>(&'a self, w: &'a Work, layer: usize) -> Vec<Call<'a>> {
        let k = &self.kernels;
        let (n, np) = (w.tokens as u32, w.padded as u32);
        let first = layer * VECTORS;
        let norm = |weight: usize| Call::Shader {
            kernel: &k.norm,
            buffers: vec![&w.x, &self.vectors, &w.h],
            constants: vec![n, WIDTH as u32, (first + weight) as u32],
            groups: [np, 1, 1],
        };
        let quantize = |width: usize| Call::Shader {
            kernel: &k.rotate8,
            buffers: vec![&w.h, &w.bytes, &w.scales],
            constants: vec![width as u32, (width / 256).div_ceil(16) as u32, 0],
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
            constants: vec![n, WIDTH as u32, (first + scale) as u32],
            groups: [(WIDTH / 4 / 64) as u32, n, 1],
        };
        let mut calls = vec![norm(NORM1), quantize(WIDTH)];
        product(&mut calls, QKV_ROWS, WIDTH, QKV_AT);
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
                (first + SCALE_QKV) as u32,
                (first + QUERY_NORM) as u32,
                (first + KEY_NORM) as u32,
            ],
            groups: [(HEADS + 2 * KV_HEADS) as u32, np, 1],
        });
        calls.push(Call::Shader {
            kernel: &k.attend,
            buffers: vec![&w.query, &w.key, &w.value, &w.h],
            constants: vec![n, np],
            groups: [np / 4, HEADS as u32, 1],
        });
        calls.push(quantize(OUT_COLUMNS));
        product(&mut calls, WIDTH, OUT_COLUMNS, OUT_AT);
        calls.push(residual(SCALE_OUT));
        calls.push(norm(NORM2));
        calls.push(quantize(WIDTH));
        product(&mut calls, GATE_ROWS, WIDTH, GATE_AT);
        calls.push(Call::Shader {
            kernel: &k.gate,
            buffers: vec![&w.sums, &w.scales, &self.vectors, &w.h],
            constants: vec![n, (first + SCALE_GATE) as u32],
            groups: [(INNER / 4 / 64) as u32, np, 1],
        });
        calls.push(quantize(INNER));
        product(&mut calls, WIDTH, INNER, DOWN_AT);
        calls.push(residual(SCALE_DOWN));
        calls
    }
}

/// Reads one layer's matrices into `to` in the window's layout.
fn read_layer(encoder: &TextEncoder, layer: usize, to: &mut [u8]) -> Result<(), &'static str> {
    let base = format!("model.layers.{layer}");
    let parts: [(&str, usize, usize); 7] = [
        ("self_attn.q_proj", HEADS * HEAD, WIDTH),
        ("self_attn.k_proj", KV_HEADS * HEAD, WIDTH),
        ("self_attn.v_proj", KV_HEADS * HEAD, WIDTH),
        ("self_attn.o_proj", WIDTH, OUT_COLUMNS),
        ("mlp.gate_proj", INNER, WIDTH),
        ("mlp.up_proj", INNER, WIDTH),
        ("mlp.down_proj", WIDTH, INNER),
    ];
    let mut rest = &mut to[..LAYER_BYTES];
    for (name, rows, columns) in parts {
        let (matrix, tail) = rest.split_at_mut(rows * columns);
        encoder
            .file
            .i8_into(&format!("{base}.{name}.weight"), &[rows, columns], matrix)?;
        rest = tail;
    }
    Ok(())
}

/// One layer's weight scales and norm weights in the `VECTORS` layout.
fn layer_vectors(encoder: &TextEncoder, layer: usize, to: &mut [f32]) -> Result<(), &'static str> {
    let file = &encoder.file;
    let base = format!("model.layers.{layer}");
    let scale = |name: &str, rows: usize| file.f32(&format!("{base}.{name}.weight_scale"), &[rows, 1]);
    let norm = |name: &str, width: usize| file.floats(&format!("{base}.{name}.weight"), &[width]);
    let parts = [
        (SCALE_QKV, scale("self_attn.q_proj", HEADS * HEAD)?),
        (SCALE_QKV + HEADS * HEAD, scale("self_attn.k_proj", KV_HEADS * HEAD)?),
        (
            SCALE_QKV + (HEADS + KV_HEADS) * HEAD,
            scale("self_attn.v_proj", KV_HEADS * HEAD)?,
        ),
        (SCALE_OUT, scale("self_attn.o_proj", WIDTH)?),
        (SCALE_GATE, scale("mlp.gate_proj", INNER)?),
        (SCALE_GATE + INNER, scale("mlp.up_proj", INNER)?),
        (SCALE_DOWN, scale("mlp.down_proj", WIDTH)?),
        (NORM1, norm("input_layernorm", WIDTH)?),
        (NORM2, norm("post_attention_layernorm", WIDTH)?),
        (QUERY_NORM, norm("self_attn.q_norm", HEAD)?),
        (KEY_NORM, norm("self_attn.k_norm", HEAD)?),
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
                LAYER_BYTES as f64 / copy.as_secs_f64() / 1e9,
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
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
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
            let row = token * WIDTH..(token + 1) * WIDTH;
            println!("token {token}: {:.1} dB", snr(&got[row.clone()], &want[row]));
        }
    }
}
