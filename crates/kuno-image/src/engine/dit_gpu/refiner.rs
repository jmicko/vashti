//! The DiT's text refiner on the GPU: two blocks of the DiT's shape with half-precision
//! weights on the text tokens alone, without modulation or rotary turns. Linear layers are
//! the half-precision product shader with f32 sums; attention is the DiT's machine-code
//! kernel at the text's token count, so this needs Level Zero. The condition projection and
//! the final norm stay on the CPU: they are small and run once per clip.
use super::{
    super::{
        gpu::{Buffer, Gpu, Kernel, Staged, bytes_of, floats, shaders},
        xe2::{attend, zebin},
    },
    Call, DIM, HEAD, HEADS, HIDDEN, INNER, Kernels, dit, run, staged_floats,
};

// Offsets into a layer's vectors.
const NORM1: usize = 0;
const NORM2: usize = NORM1 + DIM;
const QUERY_NORM: usize = NORM2 + DIM;
const KEY_NORM: usize = QUERY_NORM + HEAD;
/// The product shader's token tile.
const TILE: usize = 128;
/// Weight rows per dispatch of the product shader.
const ROW_BLOCK: usize = 4096;

/// One block's weights: half-float rows of the four matrices, then the norm weights.
struct Layer<T> {
    qkv: T,
    out: T,
    fc1: T,
    fc2: T,
    vectors: T,
}
/// The refiner's weights in host memory and its compiled kernels.
pub struct Host {
    layers: Vec<Layer<Staged>>,
    gemm16: Kernel,
    norm16: Kernel,
    merge16s: Kernel,
    gate16: Kernel,
    add: Kernel,
}
impl Host {
    /// Reads the refiner's weights (1.5 GB as half floats) into host memory.
    pub fn read(gpu: &Gpu, model: &dit::Dit) -> Result<Self, &'static str> {
        let f = &model.file;
        let layers = (0..model.refiners())
            .map(|layer| {
                let name = |part: &str| format!("token_refiner.blocks.{layer}.{part}");
                let matrix = |part: &str, rows: usize, columns: usize| {
                    gpu.staged((rows * columns) as u64 * 2, |to| {
                        f.half_into(&name(part), &[rows, columns], to)
                    })
                };
                let mut vectors = Vec::with_capacity(KEY_NORM + HEAD);
                vectors.extend(f.floats(&name("norm1.weight"), &[DIM])?);
                vectors.extend(f.floats(&name("norm2.weight"), &[DIM])?);
                vectors.extend(f.floats(&name("attn.q_norm.weight"), &[HEAD])?);
                vectors.extend(f.floats(&name("attn.k_norm.weight"), &[HEAD])?);
                Ok(Layer {
                    qkv: matrix("attn.qkv_proj.weight", INNER * 3, DIM)?,
                    out: matrix("attn.out_proj.weight", DIM, INNER)?,
                    fc1: matrix("mlp.fc1.weight", HIDDEN * 2, DIM)?,
                    fc2: matrix("mlp.fc2.weight", DIM, HIDDEN)?,
                    vectors: staged_floats(gpu, &vectors)?,
                })
            })
            .collect::<Result<_, &'static str>>()?;
        Ok(Self {
            layers,
            gemm16: gpu.kernel(shaders::GEMM16, 3, 5, Some(16))?,
            norm16: gpu.kernel(shaders::NORM16, 3, 2, None)?,
            merge16s: gpu.kernel(shaders::MERGE16S, 3, 2, None)?,
            gate16: gpu.kernel(shaders::GATE16, 2, 1, None)?,
            add: gpu.kernel(shaders::ADD, 2, 1, None)?,
        })
    }
}
/// The refiner's weights in video memory.
pub struct Refiner {
    layers: Vec<Layer<Buffer>>,
}
impl Refiner {
    pub fn place(gpu: &Gpu, host: &Host) -> Result<Self, &'static str> {
        let layers = host
            .layers
            .iter()
            .map(|layer| {
                Ok(Layer {
                    qkv: gpu.place(&layer.qkv, false)?,
                    out: gpu.place(&layer.out, false)?,
                    fc1: gpu.place(&layer.fc1, false)?,
                    fc2: gpu.place(&layer.fc2, false)?,
                    vectors: gpu.place(&layer.vectors, false)?,
                })
            })
            .collect::<Result<_, &'static str>>()?;
        Ok(Self { layers })
    }
}
impl Host {
    /// `wide[token][..rows] = half[token][..columns] . weight`.
    fn product<'a>(
        &'a self,
        half: &'a Buffer,
        weight: &'a Buffer,
        wide: &'a Buffer,
        padded: usize,
        rows: usize,
        columns: usize,
    ) -> Vec<Call<'a>> {
        (0..rows)
            .step_by(ROW_BLOCK)
            .map(|first| Call::Shader {
                kernel: &self.gemm16,
                buffers: vec![half, weight, wide],
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
            .collect()
    }
    /// `dit::Dit::refine` on the GPU: text encoder features `[tokens][TEXT_WIDTH]` to the
    /// DiT's text rows `[tokens][DIM]`. Buffers live for the call only.
    pub fn refine(
        &self,
        placed: &Refiner,
        gpu: &Gpu,
        kernels: &Kernels,
        model: &dit::Dit,
        context: &[f32],
    ) -> Result<Vec<f32>, &'static str> {
        let fast = kernels.fast.as_ref().ok_or("gpu_unsupported")?;
        let mut x = model.condition(context)?;
        let tokens = x.len() / DIM;
        if tokens == 0 {
            return Ok(x);
        }
        let padded = tokens.next_multiple_of(TILE);
        let (n, np) = (tokens as u32, padded as u32);
        x.resize(padded * DIM, 0.0);
        let stream = gpu.buffer((padded * DIM) as u64 * 4)?;
        gpu.upload(&stream, 0, bytes_of(&x))?;
        drop(x);
        // Half-float inputs of the products (up to HIDDEN wide) and their f32 sums (up to
        // 2 * HIDDEN wide).
        let half = gpu.buffer((padded * HIDDEN) as u64 * 2)?;
        let wide = gpu.buffer((padded * HIDDEN * 2) as u64 * 4)?;
        // Queries, keys with values, mixed values and weight sums in the attention kernel's
        // layouts.
        let half_rows = (HEADS * padded * HEAD) as u64 * 2;
        let sizes = [
            half_rows,
            half_rows * 2 + attend::SLACK as u64,
            half_rows * 2,
            (HEADS * padded) as u64 * 4,
        ];
        let attention = [
            gpu.shared(sizes[0])?,
            gpu.shared(sizes[1])?,
            gpu.shared(sizes[2])?,
            gpu.shared(sizes[3])?,
        ];
        let streams = (0..4)
            .map(|i| fast.import(&gpu.export(&attention[i])?, sizes[i]))
            .collect::<Result<Vec<_>, _>>()?;
        let shape = attend::Shape::new(padded, tokens, HEADS, 1.0 / (HEAD as f32).sqrt())
            .ok_or("gpu_unsupported")?;
        let native = zebin::write(&zebin::Kernel {
            name: "attend",
            code: &attend::kernel(&shape),
            grf: 256,
            pointers: 4,
        });
        let attend = fast.kernel(&native, "attend", 4)?;
        let [query, key, mixed, weights] = &attention;

        let product =
            |weight, rows, columns| self.product(&half, weight, &wide, padded, rows, columns);
        for layer in &placed.layers {
            let v = &layer.vectors;
            let norm = |weight: usize| Call::Shader {
                kernel: &self.norm16,
                buffers: vec![&stream, v, &half],
                constants: vec![n, weight as u32],
                groups: [np, 1, 1],
            };
            let add = || Call::Shader {
                kernel: &self.add,
                buffers: vec![&stream, &wide],
                constants: vec![n],
                groups: [(DIM / 4 / 64) as u32, np, 1],
            };
            let mut calls = vec![norm(NORM1)];
            calls.extend(product(&layer.qkv, INNER * 3, DIM));
            // The DiT's operand writers in their f32 form; the scale and rotary bindings
            // are not read then.
            for (side, to, weight) in [(0, query, QUERY_NORM), (1, key, KEY_NORM)] {
                calls.push(Call::Shader {
                    kernel: &kernels.qk8s,
                    buffers: vec![&wide, &stream, v, &stream, to],
                    constants: vec![n, np, side, 0, weight as u32, 1],
                    groups: [(HEADS / 4) as u32, np, 1],
                });
            }
            calls.push(Call::Shader {
                kernel: &kernels.value8s,
                buffers: vec![&wide, &stream, v, key],
                constants: vec![n, np, 0, 1],
                groups: [HEADS as u32, np / 32, 1],
            });
            calls.push(Call::Product {
                fast,
                kernel: &attend,
                memory: streams.iter().collect(),
                groups: shape.groups(HEADS),
            });
            calls.push(Call::Shader {
                kernel: &self.merge16s,
                buffers: vec![mixed, weights, &half],
                constants: vec![n, np],
                groups: [HEADS as u32, np, 1],
            });
            calls.extend(product(&layer.out, DIM, INNER));
            calls.push(add());
            calls.push(norm(NORM2));
            calls.extend(product(&layer.fc1, HIDDEN * 2, DIM));
            calls.push(Call::Shader {
                kernel: &self.gate16,
                buffers: vec![&wide, &half],
                constants: vec![n],
                groups: [(HIDDEN / 4 / 64) as u32, np, 1],
            });
            calls.extend(product(&layer.fc2, DIM, HIDDEN));
            calls.push(add());
            run(gpu, &calls)?;
        }
        let x = floats(&gpu.download(&stream, 0, tokens * DIM * 4)?);
        model.refined_norm(&x)
    }
}
