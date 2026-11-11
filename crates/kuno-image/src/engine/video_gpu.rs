//! MiniMax H3 video decoder on the GPU. Same model as `video_vae` (the CPU reference), run
//! with the Vulkan kernels in `gpu`: half-precision operands, f32 accumulation, and the
//! residual stream kept in f32.
use std::path::Path;

use rayon::prelude::*;

use super::{
    gpu::{Buffer, Gpu, Kernel, Readback, Step, bytes_of, shaders},
    safetensors::Safetensors,
    video_vae::{
        DIM, EXTRA, HEAD, HEADS, INNER, LATENT_CHANNELS, LAYERS, PAIRS, PIXELS, REGISTERS, linear,
        rotary, unpatchify,
    },
};

/// Offsets of one layer's small vectors inside its f32 vector buffer.
const NORM1: usize = 0;
const QKV_BIAS: usize = NORM1 + DIM;
const OUT_BIAS: usize = QKV_BIAS + DIM * 3;
const SCALE1: usize = OUT_BIAS + DIM;
const NORM2: usize = SCALE1 + DIM;
const W1_BIAS: usize = NORM2 + DIM;
const W2_BIAS: usize = W1_BIAS + INNER * 2;
const SCALE2: usize = W2_BIAS + DIM;
const VECTORS: usize = SCALE2 + DIM;
/// The matrix kernel works on 128-token tiles.
const TILE: usize = 128;
/// Attention heads processed per dispatch.
const GROUP: usize = 8;

struct Layer {
    qkv: Buffer,
    out: Buffer,
    w1: Buffer,
    w2: Buffer,
    vectors: Buffer,
}
struct Kernels {
    gemm: Kernel,
    gemm16: Kernel,
    mix: Kernel,
    norm: Kernel,
    heads: Kernel,
    merge: Kernel,
    residual: Kernel,
    gate: Kernel,
}
/// Scratch buffers for one padded token count, kept between tiles.
struct Work {
    padded: usize,
    stream: Buffer,
    turns: Buffer,
    half: Buffer,
    wide: Buffer,
    query: Buffer,
    key: Buffer,
    value: Buffer,
    scores: Buffer,
    mixed: Buffer,
    readback: Readback,
}
pub struct GpuVideoDecoder {
    work: std::sync::Mutex<Option<Work>>,
    gpu: Gpu,
    kernels: Kernels,
    layers: Vec<Layer>,
    /// Final norm weight and bias.
    tail: Buffer,
    project: Buffer,
    /// CPU-side ends of the model: latent to stream, and the last bias.
    post_quant: (Vec<f32>, Vec<f32>),
    embed: (Vec<f32>, Vec<f32>),
    registers: Vec<f32>,
    project_bias: Vec<f32>,
}
impl GpuVideoDecoder {
    pub fn load(gpu: &Gpu, path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        file.metadata("minimax_h3_video_vae")
            .ok_or("model_file_invalid")?;
        // Matrices go from the file straight into mapped GPU staging memory.
        let matrix = |name: &str, rows: usize, columns: usize| {
            gpu.filled((rows * columns) as u64 * 2, |to| {
                file.f16_into(name, &[rows, columns], to)
            })
        };
        let vectors = |values: &[f32]| {
            gpu.filled(values.len() as u64 * 4, |to| {
                to.copy_from_slice(bytes_of(values));
                Ok(())
            })
        };
        // Layers load in parallel: reading is the slow part and the file is in page cache.
        let layers: Result<Vec<Layer>, &'static str> = (0..LAYERS)
            .into_par_iter()
            .map(|layer| {
                let name = |part: &str| format!("decoder.transformer_blocks.{layer}.{part}");
                let mut small = Vec::with_capacity(VECTORS);
                for (part, len) in [
                    ("norm1.weight", DIM),
                    ("attn.to_qkv.bias", DIM * 3),
                    ("attn.to_out.bias", DIM),
                    ("scale1", DIM),
                    ("norm2.weight", DIM),
                    ("ff.w1.bias", INNER * 2),
                    ("ff.w2.bias", DIM),
                    ("scale2", DIM),
                ] {
                    small.extend(file.f16(&name(part), &[len])?);
                }
                Ok(Layer {
                    qkv: matrix(&name("attn.to_qkv.weight"), DIM * 3, DIM)?,
                    out: matrix(&name("attn.to_out.weight"), DIM, DIM)?,
                    w1: matrix(&name("ff.w1.weight"), INNER * 2, DIM)?,
                    w2: matrix(&name("ff.w2.weight"), DIM, INNER)?,
                    vectors: vectors(&small)?,
                })
            })
            .collect();
        let layers = layers?;
        let mut tail = file.f16("decoder.norm_out.weight", &[DIM])?;
        tail.extend(file.f16("decoder.norm_out.bias", &[DIM])?);
        let kernel = |spirv, bindings, constants| gpu.kernel(spirv, bindings, constants, None);
        Ok(Self {
            kernels: Kernels {
                gemm: gpu.kernel(shaders::GEMM, 3, 8, Some(16))?,
                gemm16: gpu.kernel(shaders::GEMM16, 3, 5, Some(16))?,
                mix: gpu.kernel(shaders::MIX, 3, 9, Some(16))?,
                norm: kernel(shaders::NORM, 3, 4)?,
                heads: kernel(shaders::HEADS, 6, 3)?,
                merge: kernel(shaders::MERGE, 2, 2)?,
                residual: kernel(shaders::RESIDUAL, 3, 3)?,
                gate: kernel(shaders::GATE, 3, 2)?,
            },
            layers,
            tail: vectors(&tail)?,
            project: matrix("decoder.proj_out.weight", PIXELS, DIM)?,
            post_quant: (
                file.f16("post_quant_conv.weight", &[24, 24, 1, 1, 1])?,
                file.f16("post_quant_conv.bias", &[24])?,
            ),
            embed: (
                file.f16("decoder.x_embedder.weight", &[DIM, 24])?,
                file.f16("decoder.x_embedder.bias", &[DIM])?,
            ),
            registers: file.f16("decoder.register_tokens", &[1, REGISTERS, DIM])?,
            project_bias: file.f16("decoder.proj_out.bias", &[PIXELS])?,
            gpu: gpu.clone(),
            work: std::sync::Mutex::new(None),
        })
    }
    /// Decodes a whole clip; see `video_tiles::decode_clip` for the layouts.
    pub fn decode(
        &self,
        latent: &[f32],
        width: usize,
        height: usize,
        frames: usize,
    ) -> Result<Vec<f32>, &'static str> {
        super::video_tiles::decode_clip(latent, width, height, frames, |tile, w, h, t| {
            self.decode_tile(tile, w, h, t)
        })
    }
    /// Same contract as `VideoDecoder::decode_tile`.
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
        let tokens = positions + EXTRA;
        let padded = tokens.div_ceil(TILE) * TILE;
        let mut start = vec![0f32; positions * LATENT_CHANNELS];
        for (channel, plane) in latent.chunks_exact(positions).enumerate() {
            for (position, value) in plane.iter().enumerate() {
                start[position * LATENT_CHANNELS + channel] = *value;
            }
        }
        let start = linear(
            &start,
            LATENT_CHANNELS,
            &self.post_quant.0,
            &self.post_quant.1,
        );
        let mut start = linear(&start, LATENT_CHANNELS, &self.embed.0, &self.embed.1);
        start.extend(&self.registers);

        let gpu = &self.gpu;
        let k = &self.kernels;
        let (n, np) = (tokens as u32, padded as u32);
        let mut cached = self.work.lock().map_err(|_| "gpu_failed")?;
        if cached.as_ref().is_none_or(|work| work.padded != padded) {
            // Drop the old set before allocating the new one.
            *cached = None;
            *cached = Some(Work {
                padded,
                stream: gpu.buffer((padded * DIM) as u64 * 4)?,
                turns: gpu.buffer((padded * PAIRS) as u64 * 8)?,
                half: gpu.buffer((padded * INNER) as u64 * 2)?,
                wide: gpu.buffer((padded * INNER * 2) as u64 * 4)?,
                query: gpu.buffer((HEADS * padded * HEAD) as u64 * 2)?,
                key: gpu.buffer((HEADS * padded * HEAD) as u64 * 2)?,
                // Rows 65..128 of each head are never written and stay zero.
                value: gpu.buffer((HEADS * TILE * padded) as u64 * 2)?,
                scores: gpu.buffer((GROUP * padded * padded) as u64 * 4)?,
                mixed: gpu.buffer((HEADS * padded * TILE) as u64 * 4)?,
                readback: gpu.readback((padded * PIXELS) as u64 * 4)?,
            });
        }
        let Some(Work {
            stream,
            turns,
            half,
            wide,
            query,
            key,
            value,
            scores,
            mixed,
            readback,
            ..
        }) = cached.as_ref()
        else {
            return Err("gpu_failed");
        };
        // Padded rows of the stream must be zero; every other buffer is rewritten in full.
        start.resize(padded * DIM, 0.0);
        gpu.upload(stream, 0, bytes_of(&start))?;
        gpu.upload(turns, 0, bytes_of(&rotary(width, height, frames)))?;
        let token_tiles = np / TILE as u32;
        type Call<'a> = (&'a Kernel, Vec<&'a Buffer>, Vec<u32>, [u32; 3]);
        // output[token][row] = input[token][..columns] . weight[row][..columns]
        // A few thousand weight rows per dispatch: the weight block then stays cached while
        // every token tile passes over it, instead of being re-read from video memory.
        let block = 4096;
        let gemm = |input, weight, output, rows: usize, columns: usize| -> Vec<Call<'_>> {
            (0..rows)
                .step_by(block)
                .map(|first| {
                    (
                        &k.gemm16,
                        vec![input, weight, output],
                        vec![
                            columns as u32,
                            0,
                            (first * columns) as u32,
                            first as u32,
                            rows as u32,
                        ],
                        [(block.min(rows - first) / 256) as u32, token_tiles, 1],
                    )
                })
                .collect()
        };
        let mut steps: Vec<Call<'_>> = Vec::new();
        for layer in &self.layers {
            let v = &layer.vectors;
            steps.push((
                &k.norm,
                vec![stream, v, half],
                vec![n, NORM1 as u32, 0, 0],
                [np, 1, 1],
            ));
            steps.extend(gemm(half, &layer.qkv, wide, DIM * 3, DIM));
            steps.push((
                &k.heads,
                vec![wide, v, turns, query, key, value],
                vec![n, np, QKV_BIAS as u32],
                [1, np, 1],
            ));
            // A few heads at a time keeps the scores small (118 MB instead of 472 MB for all
            // heads) at the same speed. The weights exist only inside `mix`.
            for head in (0..HEADS as u32).step_by(GROUP) {
                steps.push((
                    &k.gemm,
                    vec![query, key, scores],
                    vec![
                        HEAD as u32,
                        head * np * HEAD as u32,
                        np * HEAD as u32,
                        head * np * HEAD as u32,
                        np * HEAD as u32,
                        0,
                        np * np,
                        np,
                    ],
                    [token_tiles, token_tiles, GROUP as u32],
                ));
                steps.push((
                    &k.mix,
                    vec![scores, value, mixed],
                    vec![
                        np,
                        0,
                        np * np,
                        head * TILE as u32 * np,
                        TILE as u32 * np,
                        head * np * TILE as u32,
                        np * TILE as u32,
                        TILE as u32,
                        n,
                    ],
                    [1, token_tiles, GROUP as u32],
                ));
            }
            steps.push((&k.merge, vec![mixed, half], vec![n, np], [1, np, 1]));
            steps.extend(gemm(half, &layer.out, wide, DIM, DIM));
            steps.push((
                &k.residual,
                vec![stream, wide, v],
                vec![n, OUT_BIAS as u32, SCALE1 as u32],
                [(DIM / 4 / 64) as u32, np, 1],
            ));
            steps.push((
                &k.norm,
                vec![stream, v, half],
                vec![n, NORM2 as u32, 0, 0],
                [np, 1, 1],
            ));
            steps.extend(gemm(half, &layer.w1, wide, INNER * 2, DIM));
            steps.push((
                &k.gate,
                vec![wide, v, half],
                vec![n, W1_BIAS as u32],
                [(INNER / 4 / 64) as u32, np, 1],
            ));
            steps.extend(gemm(half, &layer.w2, wide, DIM, INNER));
            steps.push((
                &k.residual,
                vec![stream, wide, v],
                vec![n, W2_BIAS as u32, SCALE2 as u32],
                [(DIM / 4 / 64) as u32, np, 1],
            ));
        }
        steps.push((
            &k.norm,
            vec![stream, &self.tail, half],
            vec![n, 0, DIM as u32, 1],
            [np, 1, 1],
        ));
        steps.extend(gemm(half, &self.project, wide, PIXELS, DIM));
        let steps: Vec<Step> = steps
            .iter()
            .map(|(kernel, buffers, constants, groups)| Step {
                kernel,
                buffers,
                constants,
                groups: *groups,
            })
            .collect();
        if std::env::var_os("KUNO_GPU_PROFILE").is_some() {
            // Development aid: one submission per dispatch, summed per kernel.
            let names = [
                (&k.gemm, "gemm"),
                (&k.gemm16, "gemm16"),
                (&k.norm, "norm"),
                (&k.heads, "heads"),
                (&k.mix, "mix"),
                (&k.merge, "merge"),
                (&k.residual, "residual"),
                (&k.gate, "gate"),
            ];
            let mut totals = std::collections::BTreeMap::new();
            for step in &steps {
                let started = std::time::Instant::now();
                gpu.pass(std::slice::from_ref(step))?;
                let name = names
                    .iter()
                    .find(|n| std::ptr::eq(n.0, step.kernel))
                    .unwrap()
                    .1;
                let label = format!("{name} x{:?} c{}", step.groups, step.constants[0]);
                *totals.entry(label).or_insert(0.0) += started.elapsed().as_secs_f64();
            }
            for (label, seconds) in totals {
                eprintln!("{:8.1} ms  {label}", seconds * 1e3);
            }
        } else {
            gpu.pass(&steps)?;
        }
        let mut patches = vec![0f32; positions * PIXELS];
        gpu.read_floats(wide, 0, readback, &mut patches)?;
        patches.par_chunks_exact_mut(PIXELS).for_each(|row| {
            for (value, bias) in row.iter_mut().zip(&self.project_bias) {
                *value += bias;
            }
        });
        Ok(unpatchify(&patches, width, height, frames))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::gpu::{Select, floats},
        *,
    };

    fn read(path: &Path) -> Vec<f32> {
        floats(&std::fs::read(path).unwrap())
    }
    fn decibels(want: &[f32], got: &[f32]) -> f64 {
        assert_eq!(want.len(), got.len());
        let (mut signal, mut error) = (0f64, 0f64);
        for (w, g) in want.iter().zip(got) {
            signal += f64::from(*w).powi(2);
            error += (f64::from(*w) - f64::from(*g)).powi(2);
        }
        10.0 * (signal / error.max(f64::MIN_POSITIVE)).log10()
    }
    /// Needs the Intel GPU, the model and a captured tile (see `video_vae` tests), plus
    /// `tile-rust-cpu.f32` in the golden directory: the CPU reference's pixels for that tile,
    /// which this test writes on its first run (about half a minute).
    #[test]
    #[ignore]
    fn gpu_tile_matches_cpu_reference_and_reports_speed() {
        let model = std::env::var("KUNO_VIDEO_VAE").unwrap();
        let golden = std::path::PathBuf::from(std::env::var("KUNO_VIDEO_GOLDEN").unwrap());
        let shape: Vec<usize> = std::fs::read_to_string(golden.join("tile-shape.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height, frames) = (shape[0], shape[1], shape[2]);
        let latent = read(&golden.join("tile-latent.f32"));
        let reference = golden.join("tile-rust-cpu.f32");
        if !reference.exists() {
            let cpu = super::super::video_vae::VideoDecoder::open(Path::new(&model))
                .unwrap()
                .decode_tile(&latent, width, height, frames)
                .unwrap();
            std::fs::write(&reference, bytes_of(&cpu)).unwrap();
        }
        let (cpu, cpp) = (read(&reference), read(&golden.join("tile-pixels.f32")));
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let decoder = GpuVideoDecoder::load(&gpu, Path::new(&model)).unwrap();
        let load = started.elapsed().as_secs_f64();
        let mut seconds = Vec::new();
        let mut pixels = Vec::new();
        for _ in 0..4 {
            let started = std::time::Instant::now();
            pixels = decoder.decode_tile(&latent, width, height, frames).unwrap();
            seconds.push(started.elapsed().as_secs_f64());
        }
        let (gpu_db, cpp_db) = (decibels(&cpu, &pixels), decibels(&cpu, &cpp));
        println!(
            "load {load:.1} s; tile {seconds:.3?} s; gpu vs reference {gpu_db:.1} dB, c++ vs reference {cpp_db:.1} dB"
        );
        assert!(gpu_db >= cpp_db);
    }
    /// Needs the Intel GPU, the model and a whole decode captured from the C++ engine:
    /// `KUNO_VIDEO_CLIP=<dir>` with `clip-latent.f32`, `clip-pixels.f32` and
    /// `clip-shape.txt` ("width height latent-frames decoded-frames").
    #[test]
    #[ignore]
    fn gpu_clip_matches_cpp_engine_and_reports_speed() {
        let model = std::env::var("KUNO_VIDEO_VAE").unwrap();
        let golden = std::path::PathBuf::from(std::env::var("KUNO_VIDEO_CLIP").unwrap());
        let shape: Vec<usize> = std::fs::read_to_string(golden.join("clip-shape.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height, frames) = (shape[0], shape[1], shape[2]);
        assert_eq!(super::super::video_tiles::output_frames(frames), shape[3]);
        let latent = read(&golden.join("clip-latent.f32"));
        let cpp = read(&golden.join("clip-pixels.f32"));
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let decoder = GpuVideoDecoder::load(&gpu, Path::new(&model)).unwrap();
        let load = started.elapsed().as_secs_f64();
        let mut seconds = Vec::new();
        let mut pixels = Vec::new();
        for _ in 0..2 {
            let started = std::time::Instant::now();
            pixels = decoder.decode(&latent, width, height, frames).unwrap();
            seconds.push(started.elapsed().as_secs_f64());
        }
        let db = decibels(&cpp, &pixels);
        let worst = cpp
            .iter()
            .zip(&pixels)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "{width}x{height}x{frames} -> {} frames: load {load:.1} s, decode {seconds:.2?} s; vs c++ {db:.1} dB, largest pixel difference {worst:.4}",
            shape[3]
        );
        assert!(db > 55.0);
    }
}
