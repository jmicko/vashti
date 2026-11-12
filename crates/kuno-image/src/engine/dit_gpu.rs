//! MiniMax H3 diffusion transformer on the GPU (Vulkan). Built kernel by kernel; every piece
//! is checked against the CPU reference in `dit.rs` on identical inputs, because whole-network
//! outputs only agree loosely (see the engine document).
use std::{path::Path, sync::Mutex};

use rayon::prelude::*;

use super::{
    dit,
    gpu::{Buffer, Gpu, Kernel, Step, bytes_of, floats, shaders},
};

/// One dispatch with its arguments owned, so a pass can be assembled as a list.
pub struct Call<'a> {
    kernel: &'a Kernel,
    buffers: Vec<&'a Buffer>,
    constants: Vec<u32>,
    groups: [u32; 3],
}
/// Runs the calls in order in one submission and waits for it.
pub fn run(gpu: &Gpu, calls: &[Call]) -> Result<(), &'static str> {
    let steps: Vec<Step> = calls
        .iter()
        .map(|call| Step {
            kernel: call.kernel,
            buffers: &call.buffers,
            constants: &call.constants,
            groups: call.groups,
        })
        .collect();
    gpu.pass(&steps)
}

const DIM: usize = dit::DIM;
const INNER: usize = dit::INNER;
const HIDDEN: usize = dit::HIDDEN;
const HEADS: usize = dit::HEADS;
const HEAD: usize = dit::HEAD;
/// Offsets (in values) into a layer's vector buffer.
const NORM1: usize = 0;
const NORM2: usize = NORM1 + DIM;
const QUERY_NORM: usize = NORM2 + DIM;
const KEY_NORM: usize = QUERY_NORM + HEAD;
const SCALE_QKV: usize = KEY_NORM + HEAD;
const SCALE_OUT: usize = SCALE_QKV + INNER * 3;
const SCALE_FC1: usize = SCALE_OUT + DIM;
const SCALE_FC2: usize = SCALE_FC1 + HIDDEN * 2;
/// Tokens are padded to the matrix kernels' token tile.
const TILE: usize = 128;
/// Weight rows per dispatch of the integer product (see gemm8.comp).
const ROW_BLOCK: usize = 4096;
/// Heads per attention dispatch; bounds the score buffer.
const GROUP: usize = 8;

/// Kernels of one DiT evaluation.
pub struct Kernels {
    rotate8: Kernel,
    gemm8: Kernel,
    scoremax: Kernel,
    modnorm: Kernel,
    qk8: Kernel,
    value8: Kernel,
    rowmax: Kernel,
    mixtop: Kernel,
    merge8: Kernel,
    resid8: Kernel,
    gate8: Kernel,
}
impl Kernels {
    pub fn new(gpu: &Gpu) -> Result<Self, &'static str> {
        Ok(Self {
            rotate8: gpu.kernel(shaders::ROTATE8, 3, 2, Some(16))?,
            gemm8: gpu.kernel(shaders::GEMM8, 3, 5, Some(16))?,
            scoremax: gpu.kernel(shaders::SCOREMAX, 4, 11, Some(16))?,
            modnorm: gpu.kernel(shaders::MODNORM, 5, 5, None)?,
            qk8: gpu.kernel(shaders::QK8, 5, 5, Some(16))?,
            value8: gpu.kernel(shaders::VALUE8, 4, 3, None)?,
            rowmax: gpu.kernel(shaders::ROWMAX, 2, 4, None)?,
            mixtop: gpu.kernel(shaders::MIXTOP, 5, 12, Some(16))?,
            merge8: gpu.kernel(shaders::MERGE8, 3, 2, None)?,
            resid8: gpu.kernel(shaders::RESID8, 6, 4, None)?,
            gate8: gpu.kernel(shaders::GATE8, 4, 2, None)?,
        })
    }
    /// Quantizes `rows` f32 rows of `width` values in `input` to INT8 rows in `bytes` and one
    /// scale per row in `scales`.
    pub fn quantize<'a>(
        &'a self,
        input: &'a Buffer,
        bytes: &'a Buffer,
        scales: &'a Buffer,
        rows: usize,
        width: usize,
    ) -> Call<'a> {
        Call {
            kernel: &self.rotate8,
            buffers: vec![input, bytes, scales],
            constants: vec![width as u32, (width / 256).div_ceil(16) as u32],
            groups: [rows as u32, 1, 1],
        }
    }
    /// `sums[token][row] = bytes[token][..columns] . weight[row][..columns]` for `rows` weight
    /// rows, a block of rows per dispatch.
    fn product<'a>(
        &'a self,
        bytes: &'a Buffer,
        weight: &'a Buffer,
        sums: &'a Buffer,
        padded: usize,
        rows: usize,
        columns: usize,
    ) -> impl Iterator<Item = Call<'a>> {
        (0..rows).step_by(ROW_BLOCK).map(move |first| Call {
            kernel: &self.gemm8,
            buffers: vec![bytes, weight, sums],
            constants: vec![
                columns as u32,
                0,
                (first * columns) as u32,
                first as u32,
                rows as u32,
            ],
            groups: [
                (ROW_BLOCK.min(rows - first) / 256) as u32,
                (padded / TILE) as u32,
                1,
            ],
        })
    }
}

/// One block's weights on the GPU.
pub struct Layer {
    qkv: Buffer,
    out: Buffer,
    fc1: Buffer,
    fc2: Buffer,
    /// Norm weights and the linear layers' row scales, at the offsets above.
    vectors: Buffer,
}
impl Layer {
    pub fn load(gpu: &Gpu, model: &dit::Dit, layer: usize) -> Result<Self, &'static str> {
        let f = &model.file;
        let name = |part: &str| format!("blocks.{layer}.{part}");
        // Matrices go from the file straight into mapped GPU staging memory.
        let matrix = |part: &str, rows: usize, columns: usize| {
            gpu.filled((rows * columns) as u64, |to| {
                f.i8_into(&name(&format!("{part}.weight")), &[rows, columns], to)
            })
        };
        let mut vectors = Vec::with_capacity(SCALE_FC2 + DIM);
        vectors.extend(f.floats(&name("norm1.weight"), &[DIM])?);
        vectors.extend(f.floats(&name("norm2.weight"), &[DIM])?);
        vectors.extend(f.floats(&name("attn.q_norm.weight"), &[HEAD])?);
        vectors.extend(f.floats(&name("attn.k_norm.weight"), &[HEAD])?);
        vectors.extend(f.f32(&name("attn.qkv_proj.weight_scale"), &[INNER * 3, 1])?);
        vectors.extend(f.f32(&name("attn.out_proj.weight_scale"), &[DIM, 1])?);
        vectors.extend(f.f32(&name("mlp.fc1.weight_scale"), &[HIDDEN * 2, 1])?);
        vectors.extend(f.f32(&name("mlp.fc2.weight_scale"), &[DIM, 1])?);
        Ok(Self {
            qkv: matrix("attn.qkv_proj", INNER * 3, DIM)?,
            out: matrix("attn.out_proj", DIM, INNER)?,
            fc1: matrix("mlp.fc1", HIDDEN * 2, DIM)?,
            fc2: matrix("mlp.fc2", DIM, HIDDEN)?,
            vectors: gpu.filled(vectors.len() as u64 * 4, |to| {
                to.copy_from_slice(bytes_of(&vectors));
                Ok(())
            })?,
        })
    }
}

/// Buffers of one evaluation for a padded token count.
pub struct Work {
    pub tokens: usize,
    pub padded: usize,
    /// `[padded][DIM]` f32; padded rows are zero.
    pub stream: Buffer,
    /// Modulation row of each token.
    pub rows: Buffer,
    /// `[padded][48]` cosine and sine.
    pub turns: Buffer,
    /// Every block's modulation vectors `[blocks][rows][6][DIM]`.
    pub parts: Buffer,
    rows_count: usize,
    wide: Buffer,
    bytes: Buffer,
    scales: Buffer,
    sums: Buffer,
    query: Buffer,
    key: Buffer,
    value: Buffer,
    scores: Buffer,
    /// Largest score per key tile, `[GROUP][padded][tiles rounded up to 4]`.
    best: Buffer,
    top: Buffer,
    weights: Buffer,
    mixed: Buffer,
}
impl Work {
    pub fn new(gpu: &Gpu, tokens: usize, rows: usize, blocks: usize) -> Result<Self, &'static str> {
        let padded = tokens.next_multiple_of(TILE);
        let f32s = |count: usize| gpu.buffer(count as u64 * 4);
        Ok(Self {
            tokens,
            padded,
            stream: f32s(padded * DIM)?,
            rows: f32s(padded)?,
            turns: f32s(padded * dit::PAIRS * 2)?,
            parts: f32s(blocks * rows * 6 * DIM)?,
            rows_count: rows,
            wide: f32s(padded * HIDDEN)?,
            bytes: gpu.buffer((padded * HIDDEN) as u64)?,
            scales: f32s(padded)?,
            sums: f32s(padded * HIDDEN * 2)?,
            query: gpu.buffer((HEADS * padded * HEAD) as u64 * 2)?,
            key: gpu.buffer((HEADS * padded * HEAD) as u64 * 2)?,
            value: gpu.buffer((HEADS * HEAD * padded) as u64 * 2)?,
            scores: f32s(GROUP * padded * padded)?,
            best: f32s(GROUP * padded * (padded / TILE).next_multiple_of(4))?,
            top: f32s(HEADS * padded)?,
            weights: f32s(HEADS * padded * 8)?,
            mixed: f32s(HEADS * padded * HEAD)?,
        })
    }
}

impl Kernels {
    /// The dispatches of one block on `work.stream`; the block's modulation vectors are slot
    /// `slot` of `work.parts`.
    pub fn block<'a>(&'a self, layer: &'a Layer, work: &'a Work, slot: usize) -> Vec<Call<'a>> {
        let parts_first = (slot * work.rows_count * 6 * DIM) as u32;
        let (n, np) = (work.tokens as u32, work.padded as u32);
        let w = work;
        let v = &layer.vectors;
        let mut calls = Vec::new();
        let modnorm = |weight: usize, shift: u32, scale: u32| Call {
            kernel: &self.modnorm,
            buffers: vec![&w.stream, v, &w.parts, &w.rows, &w.wide],
            constants: vec![n, weight as u32, shift, scale, parts_first],
            groups: [np, 1, 1],
        };
        let residual = |scale: usize, gate: u32| Call {
            kernel: &self.resid8,
            buffers: vec![&w.stream, &w.sums, &w.scales, v, &w.parts, &w.rows],
            constants: vec![n, scale as u32, gate, parts_first],
            groups: [(DIM / 4 / 64) as u32, np, 1],
        };
        let quantize = |width: usize| self.quantize(&w.wide, &w.bytes, &w.scales, w.padded, width);

        calls.push(modnorm(NORM1, 0, 1));
        calls.push(quantize(DIM));
        calls.extend(self.product(&w.bytes, &layer.qkv, &w.sums, w.padded, INNER * 3, DIM));
        for (side, to, weight) in [(0, &w.query, QUERY_NORM), (1, &w.key, KEY_NORM)] {
            calls.push(Call {
                kernel: &self.qk8,
                buffers: vec![&w.sums, &w.scales, v, &w.turns, to],
                constants: vec![n, np, side, SCALE_QKV as u32, weight as u32],
                groups: [(HEADS / 4) as u32, np, 1],
            });
        }
        calls.push(Call {
            kernel: &self.value8,
            buffers: vec![&w.sums, &w.scales, v, &w.value],
            constants: vec![n, np, SCALE_QKV as u32],
            groups: [HEADS as u32, np / 32, 1],
        });
        let tiles = np / TILE as u32;
        let (head_width, group) = (HEAD as u32, GROUP as u32);
        let best_stride = tiles.next_multiple_of(4);
        for head in (0..HEADS as u32).step_by(GROUP) {
            calls.push(Call {
                kernel: &self.scoremax,
                buffers: vec![&w.query, &w.key, &w.scores, &w.best],
                constants: vec![
                    head_width,
                    head * np * head_width,
                    np * head_width,
                    head * np * head_width,
                    np * head_width,
                    0,
                    np * np,
                    np,
                    n,
                    best_stride,
                    np * best_stride,
                ],
                groups: [tiles, tiles, group],
            });
            calls.push(Call {
                kernel: &self.rowmax,
                buffers: vec![&w.best, &w.top],
                constants: vec![tiles, best_stride, np, head * np],
                groups: [1, np, group],
            });
            calls.push(Call {
                kernel: &self.mixtop,
                buffers: vec![&w.scores, &w.value, &w.mixed, &w.top, &w.weights],
                constants: vec![
                    np,
                    0,
                    np * np,
                    head * head_width * np,
                    head_width * np,
                    head * np * head_width,
                    np * head_width,
                    head_width,
                    n,
                    head * np,
                    np,
                    (1.0 / (HEAD as f32).sqrt()).to_bits(),
                ],
                groups: [1, tiles, group],
            });
        }
        calls.push(Call {
            kernel: &self.merge8,
            buffers: vec![&w.mixed, &w.weights, &w.wide],
            constants: vec![n, np],
            groups: [HEADS as u32, np, 1],
        });
        calls.push(quantize(INNER));
        calls.extend(self.product(&w.bytes, &layer.out, &w.sums, w.padded, DIM, INNER));
        calls.push(residual(SCALE_OUT, 2));

        calls.push(modnorm(NORM2, 3, 4));
        calls.push(quantize(DIM));
        calls.extend(self.product(&w.bytes, &layer.fc1, &w.sums, w.padded, HIDDEN * 2, DIM));
        calls.push(Call {
            kernel: &self.gate8,
            buffers: vec![&w.sums, &w.scales, v, &w.wide],
            constants: vec![n, SCALE_FC1 as u32],
            groups: [(HIDDEN / 4 / 64) as u32, np, 1],
        });
        calls.push(quantize(HIDDEN));
        calls.extend(self.product(&w.bytes, &layer.fc2, &w.sums, w.padded, DIM, HIDDEN));
        calls.push(residual(SCALE_FC2, 5));
        calls
    }
}

/// The whole DiT with its weights on the GPU.
pub struct GpuDit {
    gpu: Gpu,
    model: dit::Dit,
    kernels: Kernels,
    layers: Vec<Layer>,
    work: Mutex<Option<Work>>,
}
impl GpuDit {
    /// Reads all blocks (about 19 GB) into video memory.
    pub fn load(gpu: &Gpu, path: &Path) -> Result<Self, &'static str> {
        let model = dit::Dit::open(path)?;
        let kernels = Kernels::new(gpu)?;
        // A few layers at a time: reading is the slow part, and each matrix passes through
        // host staging memory of its own size.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .map_err(|_| "gpu_failed")?;
        let layers: Result<Vec<Layer>, &'static str> = pool.install(|| {
            (0..model.layers)
                .into_par_iter()
                .map(|layer| Layer::load(gpu, &model, layer))
                .collect()
        });
        Ok(Self {
            gpu: gpu.clone(),
            model,
            kernels,
            layers: layers?,
            work: Mutex::new(None),
        })
    }
    /// Evaluates the network once; returns the packed velocity with `step.packed`'s layout.
    /// Embedding, the text refiner and the output heads still run on the CPU.
    pub fn forward(&self, step: &dit::Step) -> Result<Vec<f32>, &'static str> {
        let prepared = self.model.prepare(step)?;
        self.blocks(step, prepared)
    }
    fn blocks(&self, step: &dit::Step, prepared: dit::Prepared) -> Result<Vec<f32>, &'static str> {
        let gpu = &self.gpu;
        let dit::Prepared {
            mut x,
            layout,
            times,
            rotary,
        } = prepared;
        let tokens = layout.positions.len();
        let rows = layout.timesteps.len() * 3;
        let mut cached = self.work.lock().map_err(|_| "gpu_failed")?;
        if cached
            .as_ref()
            .is_none_or(|w| w.padded != tokens.next_multiple_of(TILE) || w.rows_count != rows)
        {
            // Drop the old set before allocating the new one.
            *cached = None;
            *cached = Some(Work::new(gpu, tokens, rows, self.layers.len())?);
        }
        let work = cached.as_mut().ok_or("gpu_failed")?;
        work.tokens = tokens;
        let mut token_rows = vec![0u32; work.padded];
        for span in &layout.spans {
            token_rows[span.start..span.end].fill(span.row as u32);
        }
        let mut turns: Vec<f32> = rotary.iter().flatten().copied().collect();
        turns.resize(work.padded * dit::PAIRS * 2, 0.0);
        let mut parts = Vec::with_capacity(self.layers.len() * rows * 6 * DIM);
        for layer in 0..self.layers.len() {
            parts.extend(self.model.parts(layer, &times)?);
        }
        // Padded rows of the stream must be zero; every other buffer is rewritten in full.
        x.resize(work.padded * DIM, 0.0);
        gpu.upload(&work.stream, 0, bytes_of(&x))?;
        gpu.upload(&work.rows, 0, bytes_of(&token_rows))?;
        gpu.upload(&work.turns, 0, bytes_of(&turns))?;
        gpu.upload(&work.parts, 0, bytes_of(&parts))?;
        let work = &*work;
        // One submission per block: the kernel driver ends jobs that run for several seconds.
        for (slot, layer) in self.layers.iter().enumerate() {
            run(gpu, &self.kernels.block(layer, work, slot))?;
        }
        let x = floats(&gpu.download(&work.stream, 0, tokens * DIM * 4)?);
        self.model.finish(step, &x, &layout, &times)
    }
}

#[cfg(test)]
mod tests {
    use super::{super::gpu::Select, *};

    fn noise(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    #[ignore = "needs the Intel GPU"]
    fn quantizer_matches_reference_bit_for_bit() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let kernels = Kernels::new(&gpu).unwrap();
        for width in [5376usize, 7168, 14336] {
            let rows = 37;
            let mut x = noise(rows * width, width as u64);
            // Uneven magnitudes, a zero row and exact rounding ties.
            for (i, v) in x.iter_mut().enumerate() {
                *v *= 1.0 + (i % 97) as f32;
            }
            x[3 * width..4 * width].fill(0.0);
            for (i, v) in x[5 * width..6 * width].iter_mut().enumerate() {
                *v = (i % 5) as f32 * 0.5;
            }
            let (want, want_scales) = dit::quantize(&x, width);
            let input = gpu.buffer((rows * width) as u64 * 4).unwrap();
            let bytes = gpu.buffer((rows * width) as u64).unwrap();
            let scales = gpu.buffer(rows as u64 * 4).unwrap();
            gpu.upload(&input, 0, bytes_of(&x)).unwrap();
            run(
                &gpu,
                &[kernels.quantize(&input, &bytes, &scales, rows, width)],
            )
            .unwrap();
            let got = gpu.download(&bytes, 0, rows * width).unwrap();
            let got_scales = floats(&gpu.download(&scales, 0, rows * 4).unwrap());
            assert_eq!(got_scales, want_scales, "width {width}");
            let differ = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| **a as i8 != **b)
                .count();
            assert_eq!(differ, 0, "width {width}");

            let rows = 8704;
            let input = gpu.buffer((rows * width) as u64 * 4).unwrap();
            let bytes = gpu.buffer((rows * width) as u64).unwrap();
            let scales = gpu.buffer(rows as u64 * 4).unwrap();
            let call = [kernels.quantize(&input, &bytes, &scales, rows, width)];
            run(&gpu, &call).unwrap();
            let started = std::time::Instant::now();
            for _ in 0..4 {
                run(&gpu, &call).unwrap();
            }
            println!(
                "{rows} rows of {width}: {:.2} ms",
                started.elapsed().as_secs_f64() * 250.0
            );
        }
    }

    /// One block on the GPU against the reference on the same input, from a captured step.
    #[test]
    #[ignore = "needs the Intel GPU, the model and a captured step"]
    fn block_matches_reference() {
        let (Ok(model), Ok(golden)) = (std::env::var("KUNO_DIT"), std::env::var("KUNO_DIT_GOLDEN"))
        else {
            panic!("set KUNO_DIT and KUNO_DIT_GOLDEN (a captured step's path prefix)");
        };
        let captured = dit::tests::Captured::read(&golden);
        let model = dit::Dit::open(std::path::Path::new(&model)).unwrap();
        let prepared = model.prepare(&captured.step()).unwrap();
        let tokens = prepared.layout.positions.len();
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let kernels = Kernels::new(&gpu).unwrap();
        let rows = prepared.layout.timesteps.len() * 3;
        let work = Work::new(&gpu, tokens, rows, 1).unwrap();
        let mut token_rows = vec![0u32; work.padded];
        for span in &prepared.layout.spans {
            token_rows[span.start..span.end].fill(span.row as u32);
        }
        let mut turns: Vec<f32> = prepared.rotary.iter().flatten().copied().collect();
        turns.resize(work.padded * dit::PAIRS * 2, 0.0);
        gpu.upload(&work.rows, 0, bytes_of(&token_rows)).unwrap();
        gpu.upload(&work.turns, 0, bytes_of(&turns)).unwrap();
        for index in [0usize, 20, 49] {
            let layer = Layer::load(&gpu, &model, index).unwrap();
            let parts = model.parts(index, &prepared.times).unwrap();
            let mut want = prepared.x.clone();
            model
                .block(
                    index,
                    &mut want,
                    &parts,
                    &prepared.layout.spans,
                    &prepared.rotary,
                )
                .unwrap();
            let mut start = prepared.x.clone();
            start.resize(work.padded * DIM, 0.0);
            gpu.upload(&work.stream, 0, bytes_of(&start)).unwrap();
            gpu.upload(&work.parts, 0, bytes_of(&parts)).unwrap();
            let calls = kernels.block(&layer, &work, 0);
            let started = std::time::Instant::now();
            run(&gpu, &calls).unwrap();
            let seconds = started.elapsed().as_secs_f64();
            let got = floats(&gpu.download(&work.stream, 0, tokens * DIM * 4).unwrap());
            let (mut error, mut change) = (0f64, 0f64);
            for ((got, want), before) in got.iter().zip(&want).zip(&prepared.x) {
                error += f64::from(got - want).powi(2);
                change += f64::from(want - before).powi(2);
            }
            let db = 10.0 * (change / error.max(1e-30)).log10();
            println!(
                "block {index}: {db:.1} dB from the reference's update, {tokens} tokens, {:.1} ms",
                seconds * 1e3
            );
            assert!(db > 30.0);
        }
    }

    /// Time of one block at a real clip's size (2.3 s at 832x544 is 8611 tokens).
    #[test]
    #[ignore = "needs the Intel GPU and the model"]
    fn block_speed_at_clip_size() {
        let model =
            dit::Dit::open(std::path::Path::new(&std::env::var("KUNO_DIT").unwrap())).unwrap();
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let kernels = Kernels::new(&gpu).unwrap();
        let (tokens, rows) = (8611, 12);
        let work = Work::new(&gpu, tokens, rows, 1).unwrap();
        let mut start = noise(tokens * DIM, 5);
        start.resize(work.padded * DIM, 0.0);
        gpu.upload(&work.stream, 0, bytes_of(&start)).unwrap();
        gpu.upload(&work.rows, 0, bytes_of(&vec![0u32; work.padded]))
            .unwrap();
        let mut turns = vec![[1f32, 0.0]; work.padded * dit::PAIRS];
        for (i, turn) in turns.iter_mut().enumerate() {
            let angle = i as f32 * 0.01;
            *turn = [angle.cos(), angle.sin()];
        }
        gpu.upload(&work.turns, 0, bytes_of(&turns)).unwrap();
        gpu.upload(&work.parts, 0, bytes_of(&noise(rows * 6 * DIM, 6)))
            .unwrap();
        let layer = Layer::load(&gpu, &model, 10).unwrap();
        let calls = kernels.block(&layer, &work, 0);
        run(&gpu, &calls).unwrap();
        let names = [
            (&kernels.rotate8, "quantize"),
            (&kernels.gemm8, "linear"),
            (&kernels.scoremax, "scores"),
            (&kernels.modnorm, "norm"),
            (&kernels.qk8, "queries and keys"),
            (&kernels.value8, "values"),
            (&kernels.rowmax, "row maximum"),
            (&kernels.mixtop, "mixing"),
            (&kernels.merge8, "merge"),
            (&kernels.resid8, "residual"),
            (&kernels.gate8, "gate"),
        ];
        let mut totals = std::collections::BTreeMap::new();
        for call in &calls {
            let started = std::time::Instant::now();
            run(&gpu, std::slice::from_ref(call)).unwrap();
            let name = names
                .iter()
                .find(|n| std::ptr::eq(n.0, call.kernel))
                .unwrap()
                .1;
            *totals.entry(name).or_insert(0.0) += started.elapsed().as_secs_f64();
        }
        for (name, seconds) in &totals {
            println!("{:8.1} ms  {name}", seconds * 1e3);
        }
        let started = std::time::Instant::now();
        run(&gpu, &calls).unwrap();
        println!(
            "one block, {tokens} tokens: {:.1} ms in one submission",
            started.elapsed().as_secs_f64() * 1e3
        );
    }

    /// The whole network on the GPU against a step captured from the C++ engine.
    #[test]
    #[ignore = "needs the Intel GPU with 25 GB free, the model and a captured step"]
    fn step_matches_cpp_engine() {
        let (Ok(model), Ok(golden)) = (std::env::var("KUNO_DIT"), std::env::var("KUNO_DIT_GOLDEN"))
        else {
            panic!("set KUNO_DIT and KUNO_DIT_GOLDEN (a captured step's path prefix)");
        };
        let captured = dit::tests::Captured::read(&golden);
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let network = GpuDit::load(&gpu, std::path::Path::new(&model)).unwrap();
        println!("loaded in {:.1} s", started.elapsed().as_secs_f64());
        let step = captured.step();
        let mut got = Vec::new();
        for _ in 0..2 {
            let prepared = network.model.prepare(&step).unwrap();
            let started = std::time::Instant::now();
            got = network.blocks(&step, prepared).unwrap();
            println!("blocks and heads: {:.2} s", started.elapsed().as_secs_f64());
        }
        let plane = captured.shape[0] * captured.shape[1] * captured.shape[2];
        let report = |name: &str, range: std::ops::Range<usize>| {
            let (mut error, mut power) = (0f64, 0f64);
            for i in range {
                error += f64::from(got[i] - captured.out[i]).powi(2);
                power += f64::from(captured.out[i]).powi(2);
            }
            let db = 10.0 * (power / error.max(1e-30)).log10();
            println!("{name}: {db:.1} dB from the C++ engine");
            db
        };
        let video = report("video", 0..plane * dit::VIDEO_CHANNELS);
        let audio = report(
            "audio",
            plane * dit::VIDEO_CHANNELS
                ..plane * dit::VIDEO_CHANNELS + captured.audio_length * 2 * dit::AUDIO_CHANNELS,
        );
        // Whole-network agreement is loose by nature (see dit.rs) and looser with more tokens:
        // 29 and 24 dB on a 317-token step, 24 and 18 dB on an 8611-token one whose single
        // blocks agree with the reference to 40 dB and better. These bounds catch structure.
        assert!(video > 20.0 && audio > 15.0);
    }

    /// Replays a captured clip's sampler steps: every step from the C++ engine's input, and
    /// the whole Euler loop from the first noise on this DiT's own outputs. Optionally
    /// decodes both final latents and compares frames.
    #[test]
    #[ignore = "needs the Intel GPU with 25 GB free, the model and a captured clip"]
    fn sampling_follows_cpp_engine() {
        let model = std::env::var("KUNO_DIT").unwrap();
        let dir = std::env::var("KUNO_DIT_STEPS").unwrap();
        let steps: Vec<_> = (0..)
            .map(|i| format!("{dir}/step{i}"))
            .take_while(|prefix| std::path::Path::new(&format!("{prefix}.txt")).exists())
            .map(|prefix| dit::tests::Captured::read(&prefix))
            .collect();
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let network = GpuDit::load(&gpu, std::path::Path::new(&model)).unwrap();
        let decibels = |got: &[f32], want: &[f32]| {
            let (mut error, mut power) = (0f64, 0f64);
            for (got, want) in got.iter().zip(want) {
                error += f64::from(got - want).powi(2);
                power += f64::from(*want).powi(2);
            }
            10.0 * (power / error.max(1e-30)).log10()
        };
        let sigma = |i: usize| steps.get(i).map_or(0.0, |s| s.timestep / 1000.0);
        let euler = |x: &[f32], out: &[f32], i: usize| -> Vec<f32> {
            let change = sigma(i + 1) - sigma(i);
            x.iter().zip(out).map(|(x, v)| x + change * v).collect()
        };
        let mut own = steps[0].x.clone();
        for (i, captured) in steps.iter().enumerate() {
            let out = network.forward(&captured.step()).unwrap();
            let mut step = captured.step();
            step.packed = &own;
            let next = euler(&own, &network.forward(&step).unwrap(), i);
            println!(
                "step {i} (sigma {:.3}): output {:.1} dB from the C++ engine; own loop's latent {:.1} dB",
                sigma(i),
                decibels(&out, &captured.out),
                decibels(&own, &captured.x),
            );
            own = next;
        }
        let last = steps.len() - 1;
        let theirs = euler(&steps[last].x, &steps[last].out, last);
        let shape = &steps[0].shape;
        let video = shape[0] * shape[1] * shape[2] * dit::VIDEO_CHANNELS;
        println!(
            "final video latent: {:.1} dB from the C++ engine's",
            decibels(&own[..video], &theirs[..video])
        );
        let Ok(vae) = std::env::var("KUNO_VIDEO_VAE") else {
            return;
        };
        drop(network);
        let decoder =
            super::super::video_gpu::GpuVideoDecoder::load(&gpu, std::path::Path::new(&vae))
                .unwrap();
        let decode = |latent: &[f32]| {
            decoder
                .decode(
                    &dit::decoder_latent(&latent[..video]),
                    shape[0],
                    shape[1],
                    shape[2],
                )
                .unwrap()
        };
        let (mine, theirs) = (decode(&own), decode(&theirs));
        let mut error = 0f64;
        for (a, b) in mine.iter().zip(&theirs) {
            error += f64::from(a.clamp(0.0, 1.0) - b.clamp(0.0, 1.0)).powi(2);
        }
        println!(
            "decoded frames: {:.1} dB PSNR between the two",
            10.0 * (mine.len() as f64 / error.max(1e-30)).log10()
        );
        if let Ok(out) = std::env::var("KUNO_DIT_FRAMES_OUT") {
            let (width, height) = (shape[0] * 16, shape[1] * 16);
            let frames = mine.len() / (3 * width * height);
            for (name, pixels) in [("rust", &mine), ("cpp", &theirs)] {
                for frame in [0, frames / 2, frames - 1] {
                    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
                    for at in 0..width * height {
                        for c in 0..3 {
                            let v = pixels[(c * frames + frame) * width * height + at];
                            ppm.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                        }
                    }
                    std::fs::write(format!("{out}/{name}-{frame}.ppm"), ppm).unwrap();
                }
            }
        }
    }
}
