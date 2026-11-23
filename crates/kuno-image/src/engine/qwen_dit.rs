//! Qwen Image 2.1's diffusion transformer on the CPU: the reference the GPU form is checked
//! against, written from the C++ engine's graph (`qwen_image_2_1.hpp`).
//!
//! One stream of 4096-wide tokens: the prompt's text tokens (with each reference image's
//! latent rows in place of its vision tokens), then the target latent. 32 blocks of
//! LayerNorm (no weights) scaled by `1 + m`, attention with RMS-normed queries and keys and
//! three-axis rotary positions, and a SwiGLU MLP; residual adds are gated by `tanh(m)`.
//! The prefix (everything before the target) is modulated with the time-zero row and never
//! attends to the target, so its keys and values do not change between steps. Text tokens
//! attend causally within their run and to everything before it; image tokens attend to
//! everything up to their own end. All block linears are INT8 with Hadamard rotation.

use std::path::Path;

use rayon::prelude::*;

use super::{
    dit::{linear, linear8},
    safetensors::Safetensors,
};

pub const CHANNELS: usize = 64;
pub const CONTEXT: usize = 4096;
pub(super) const DIM: usize = 4096;
pub(super) const HEAD: usize = 128;
pub(super) const HEADS: usize = DIM / HEAD;
pub(super) const HIDDEN: usize = 12288;
pub(super) const PAIRS: usize = HEAD / 2;
const TIME: usize = 256;
const AXES: [usize; 3] = [16, 56, 56];
const EPS: f32 = 1e-6;

/// A run of tokens in the joint stream: text (`image == None`) or one image's latent rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
    /// First context row of a text run, or the image's slot run in the context.
    pub context_start: usize,
    /// Reference index, or `references.len()` for the target.
    pub image: Option<usize>,
}

/// Token runs, positions and the prefix length for one prompt and set of images.
#[derive(Clone, Debug)]
pub struct Layout {
    pub segments: Vec<Segment>,
    pub positions: Vec<[f32; 3]>,
    pub prefix: usize,
}
impl Layout {
    /// `slots` tags each context row: 0 for text, `i + 1` for reference `i`'s vision tokens
    /// (empty means all text). `images` is each reference's latent `(height, width)`, then
    /// the target's. A reference's slot run must hold a quarter of its latent cells.
    pub fn new(
        text: usize,
        slots: &[i32],
        images: &[(usize, usize)],
    ) -> Result<Self, &'static str> {
        if images.is_empty() || (!slots.is_empty() && slots.len() != text) {
            return Err("invalid_input");
        }
        let mut layout = Layout {
            segments: Vec::new(),
            positions: Vec::new(),
            prefix: 0,
        };
        let mut position = 0f32;
        let mut next = 0;
        let append_image =
            |layout: &mut Layout, index: usize, context_start: usize, position: &mut f32| {
                let (height, width) = images[index];
                let start = layout.positions.len();
                layout.segments.push(Segment {
                    start,
                    end: start + height * width,
                    context_start,
                    image: Some(index),
                });
                let (h0, w0) = ((height - height / 2) as f32, (width - width / 2) as f32);
                for h in 0..height {
                    for w in 0..width {
                        layout
                            .positions
                            .push([*position, h as f32 - h0, w as f32 - w0]);
                    }
                }
                *position += height.max(width) as f32;
            };
        let tag = |i: usize| if slots.is_empty() { 0 } else { slots[i] };
        let mut i = 0;
        while i < text {
            let (value, begin) = (tag(i), i);
            i += 1;
            while i < text && tag(i) == value {
                i += 1;
            }
            if value != 0 {
                if value as usize != next + 1
                    || next + 1 >= images.len()
                    || (i - begin) * 4 != images[next].0 * images[next].1
                {
                    return Err("invalid_input");
                }
                append_image(&mut layout, next, begin, &mut position);
                next += 1;
            } else {
                let start = layout.positions.len();
                layout.segments.push(Segment {
                    start,
                    end: start + i - begin,
                    context_start: begin,
                    image: None,
                });
                for _ in begin..i {
                    layout.positions.push([position; 3]);
                    position += 1.0;
                }
            }
        }
        if next + 1 != images.len() {
            return Err("invalid_input");
        }
        layout.prefix = layout.positions.len();
        append_image(&mut layout, next, text, &mut position);
        Ok(layout)
    }
    pub fn tokens(&self) -> usize {
        self.positions.len()
    }
}

/// `[tokens][PAIRS]` cosine and sine, pairs `(2i, 2i + 1)`: 8 time, 28 height, 28 width
/// frequencies, computed in the C++ engine's float order.
pub(super) fn rotary(positions: &[[f32; 3]]) -> Vec<[f32; 2]> {
    let mut frequencies = Vec::with_capacity(PAIRS);
    for (axis, &dim) in AXES.iter().enumerate() {
        let half = dim / 2;
        let step = ((dim as f32 - 2.0) / dim as f32) / (half - 1) as f32;
        for j in 0..half {
            frequencies.push((axis, 1.0 / 10000f32.powf(j as f32 * step)));
        }
    }
    let mut out = Vec::with_capacity(positions.len() * PAIRS);
    for position in positions {
        for &(axis, frequency) in &frequencies {
            let angle = position[axis] * frequency;
            out.push([angle.cos(), angle.sin()]);
        }
    }
    out
}

/// Flux-style sinusoidal embedding: cosines then sines, `exp(-ln(10000) j / 128)`.
pub(super) fn time_embedding(t: f32) -> Vec<f32> {
    let half = TIME / 2;
    let mut out = vec![0f32; TIME];
    for j in 0..half {
        let frequency = (-(10000f32.ln()) * j as f32 / half as f32).exp();
        let angle = t * frequency;
        out[j] = angle.cos();
        out[j + half] = angle.sin();
    }
    out
}
fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}
pub(super) fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044_715 * v * v * v)).tanh())
}
/// Rows normalized to zero mean and unit variance (no weights).
fn layer_norm(x: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    out.par_chunks_exact_mut(DIM).for_each(|row| {
        let mean = row.iter().sum::<f32>() / DIM as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / DIM as f32;
        let scale = 1.0 / (var + EPS).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) * scale;
        }
    });
    out
}
fn rms_norm(row: &mut [f32], weight: &[f32]) {
    let mean = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let scale = 1.0 / (mean + EPS).sqrt();
    for (v, w) in row.iter_mut().zip(weight) {
        *v *= scale * w;
    }
}

/// Inputs of one evaluation.
pub struct Step<'a> {
    /// `[CHANNELS][height][width]`.
    pub x: &'a [f32],
    pub height: usize,
    pub width: usize,
    /// On the `[0, 1000]` scale.
    pub timestep: f32,
    /// Text encoder rows `[tokens][CONTEXT]`.
    pub context: &'a [f32],
    /// Per context row: 0 text, `i + 1` reference `i`; empty for all text.
    pub slots: &'a [i32],
    /// Each reference latent `[CHANNELS][height][width]` with its `(height, width)`.
    pub references: Vec<(&'a [f32], usize, usize)>,
}

/// `[CHANNELS][h][w]` to `[h * w][CHANNELS]`.
pub(super) fn patchify(latent: &[f32], height: usize, width: usize) -> Vec<f32> {
    let plane = height * width;
    let mut out = vec![0f32; plane * CHANNELS];
    for (cell, row) in out.chunks_exact_mut(CHANNELS).enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = latent[c * plane + cell];
        }
    }
    out
}
pub(super) fn unpatchify(rows: &[f32], height: usize, width: usize) -> Vec<f32> {
    let plane = height * width;
    let mut out = vec![0f32; plane * CHANNELS];
    for (cell, row) in rows.chunks_exact(CHANNELS).enumerate() {
        for (c, v) in row.iter().enumerate() {
            out[c * plane + cell] = *v;
        }
    }
    out
}

pub struct QwenDit {
    pub(super) file: Safetensors,
    pub(super) layers: usize,
}
impl QwenDit {
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        let layers = file.count("transformer_blocks.");
        if layers == 0 {
            return Err("model_file_invalid");
        }
        Ok(Self { file, layers })
    }
    fn floats(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, &'static str> {
        self.file.floats(name, shape)
    }
    fn int8(
        &self,
        name: &str,
        outputs: usize,
        inputs: usize,
    ) -> Result<(Vec<i8>, Vec<f32>), &'static str> {
        Ok((
            self.file
                .i8(&format!("{name}.weight"), &[outputs, inputs])?,
            self.file
                .f32(&format!("{name}.weight_scale"), &[outputs, 1])?,
        ))
    }
    fn dense(
        &self,
        x: &[f32],
        name: &str,
        outputs: usize,
        inputs: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let weight = self.floats(&format!("{name}.weight"), &[outputs, inputs])?;
        Ok(linear(x, inputs, &weight, &[]))
    }
    fn quantized(
        &self,
        x: &[f32],
        name: &str,
        outputs: usize,
        inputs: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let (weight, scale) = self.int8(name, outputs, inputs)?;
        Ok(linear8(x, inputs, &weight, &scale))
    }

    /// `silu(time embedding)` for `[t, 0]`: `[2][DIM]`.
    pub(super) fn times(&self, timestep: f32) -> Result<Vec<f32>, &'static str> {
        let mut rows = time_embedding(timestep);
        rows.extend(time_embedding(0.0));
        let name = "time_text_embed.timestep_embedder";
        let mut h = self.dense(&rows, &format!("{name}.linear_1"), DIM, TIME)?;
        h.iter_mut().for_each(|v| *v = silu(*v));
        let mut h = self.dense(&h, &format!("{name}.linear_2"), DIM, DIM)?;
        h.iter_mut().for_each(|v| *v = silu(*v));
        Ok(h)
    }
    /// Text encoder rows to the stream's width.
    pub(super) fn text(&self, context: &[f32]) -> Result<Vec<f32>, &'static str> {
        let weight: Vec<f32> = self
            .floats("txt_in.text_norm.weight", &[CONTEXT])?
            .iter()
            .map(|w| w + 1.0)
            .collect();
        let mut x = context.to_vec();
        x.par_chunks_exact_mut(CONTEXT)
            .for_each(|row| rms_norm(row, &weight));
        let mut h = self.dense(&x, "txt_in.in_layer", DIM, CONTEXT)?;
        h.iter_mut().for_each(|v| *v = gelu(*v));
        self.dense(&h, "txt_in.out_layer", DIM, DIM)
    }

    pub fn forward(&self, step: &Step) -> Result<Vec<f32>, &'static str> {
        let text_len = step.context.len() / CONTEXT;
        let mut images: Vec<(usize, usize)> = step.references.iter().map(|r| (r.1, r.2)).collect();
        images.push((step.height, step.width));
        let layout = Layout::new(text_len, step.slots, &images)?;
        let times = self.times(step.timestep)?;
        let modulation = self.dense(&times, "modulation.1", 4 * DIM, DIM)?;
        let text = self.text(step.context)?;
        let img_in = self.floats("img_in.weight", &[DIM, CHANNELS])?;
        let mut x = Vec::with_capacity(layout.tokens() * DIM);
        for segment in &layout.segments {
            match segment.image {
                None => x.extend_from_slice(
                    &text[segment.context_start * DIM..][..(segment.end - segment.start) * DIM],
                ),
                Some(index) => {
                    let (latent, h, w) = if index == step.references.len() {
                        (step.x, step.height, step.width)
                    } else {
                        step.references[index]
                    };
                    x.extend(linear(&patchify(latent, h, w), CHANNELS, &img_in, &[]));
                }
            }
        }
        let rotary = rotary(&layout.positions);
        for layer in 0..self.layers {
            x = self.block(layer, x, &layout, &modulation, &rotary)?;
        }
        let x = x[layout.prefix * DIM..].to_vec();
        let scale = self.dense(&times[..DIM], "norm_out.linear", DIM, DIM)?;
        let mut h = layer_norm(&x);
        h.par_chunks_exact_mut(DIM).for_each(|row| {
            for (v, s) in row.iter_mut().zip(&scale) {
                *v *= 1.0 + s;
            }
        });
        let out = self.dense(&h, "proj_out", CHANNELS, DIM)?;
        Ok(unpatchify(&out, step.height, step.width))
    }

    /// `x * (1 + m)` or, for a gate, `x * tanh(m)`, with modulation part `part` of the time
    /// row (target) or the time-zero row (prefix).
    fn modulate(x: &mut [f32], modulation: &[f32], part: usize, prefix: usize, gate: bool) {
        let row = |r: usize| &modulation[r * 4 * DIM + part * DIM..][..DIM];
        let (target, zero) = (row(0), row(1));
        x.par_chunks_exact_mut(DIM)
            .enumerate()
            .for_each(|(token, v)| {
                let m = if token < prefix { zero } else { target };
                for (v, m) in v.iter_mut().zip(m) {
                    *v *= if gate { m.tanh() } else { 1.0 + m };
                }
            });
    }

    fn block(
        &self,
        layer: usize,
        mut x: Vec<f32>,
        layout: &Layout,
        modulation: &[f32],
        rotary: &[[f32; 2]],
    ) -> Result<Vec<f32>, &'static str> {
        let name = format!("transformer_blocks.{layer}");
        let prefix = layout.prefix;
        let mut h = layer_norm(&x);
        Self::modulate(&mut h, modulation, 0, prefix, false);
        let q = self.quantized(&h, &format!("{name}.attn.to_q"), DIM, DIM)?;
        let k = self.quantized(&h, &format!("{name}.attn.to_k"), DIM, DIM)?;
        let v = self.quantized(&h, &format!("{name}.attn.to_v"), DIM, DIM)?;
        let qw = self.floats(&format!("{name}.attn.norm_q.weight"), &[HEAD])?;
        let kw = self.floats(&format!("{name}.attn.norm_k.weight"), &[HEAD])?;
        let attended = attention(&q, &k, &v, &qw, &kw, rotary, &layout.segments);
        let mut out = self.quantized(&attended, &format!("{name}.attn.to_out.0"), DIM, DIM)?;
        Self::modulate(&mut out, modulation, 1, prefix, true);
        x.iter_mut().zip(&out).for_each(|(x, o)| *x += o);
        let mut h = layer_norm(&x);
        Self::modulate(&mut h, modulation, 2, prefix, false);
        let gate_up = self.quantized(&h, &format!("{name}.img_mlp.gate_up"), 2 * HIDDEN, DIM)?;
        let mut act = vec![0f32; x.len() / DIM * HIDDEN];
        act.par_chunks_exact_mut(HIDDEN)
            .zip(gate_up.par_chunks_exact(2 * HIDDEN))
            .for_each(|(out, row)| {
                for i in 0..HIDDEN {
                    out[i] = silu(row[i]) * row[HIDDEN + i];
                }
            });
        let mut out = self.quantized(&act, &format!("{name}.img_mlp.out"), DIM, HIDDEN)?;
        Self::modulate(&mut out, modulation, 3, prefix, true);
        x.iter_mut().zip(&out).for_each(|(x, o)| *x += o);
        Ok(x)
    }
}

/// Multi-head attention over the joint stream. `q`, `k`, `v` are `[tokens][DIM]`; queries
/// and keys get the weighted RMS norm and rotary positions. Each segment's queries see keys
/// `0..end`, text runs causally.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qw: &[f32],
    kw: &[f32],
    rotary: &[[f32; 2]],
    segments: &[Segment],
) -> Vec<f32> {
    let tokens = q.len() / DIM;
    let prepare = |x: &[f32], weight: &[f32], head: usize| {
        let mut out = vec![0f32; tokens * HEAD];
        for (token, row) in out.chunks_exact_mut(HEAD).enumerate() {
            row.copy_from_slice(&x[token * DIM + head * HEAD..][..HEAD]);
            rms_norm(row, weight);
            for (pair, [cos, sin]) in rotary[token * PAIRS..][..PAIRS].iter().enumerate() {
                let (a, b) = (row[2 * pair], row[2 * pair + 1]);
                row[2 * pair] = cos * a - sin * b;
                row[2 * pair + 1] = sin * a + cos * b;
            }
        }
        out
    };
    let scale = 1.0 / (HEAD as f32).sqrt();
    let heads: Vec<Vec<f32>> = (0..HEADS)
        .into_par_iter()
        .map(|head| {
            let (query, key) = (prepare(q, qw, head), prepare(k, kw, head));
            let mut out = vec![0f32; tokens * HEAD];
            let mut scores = vec![0f32; tokens];
            for segment in segments {
                for t in segment.start..segment.end {
                    let keys = if segment.image.is_some() {
                        segment.end
                    } else {
                        t + 1
                    };
                    let qrow = &query[t * HEAD..][..HEAD];
                    let mut top = f32::NEG_INFINITY;
                    for (j, s) in scores[..keys].iter_mut().enumerate() {
                        let krow = &key[j * HEAD..][..HEAD];
                        *s = qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale;
                        top = top.max(*s);
                    }
                    let mut sum = 0f32;
                    for s in &mut scores[..keys] {
                        *s = (*s - top).exp();
                        sum += *s;
                    }
                    let row = &mut out[t * HEAD..][..HEAD];
                    for (j, s) in scores[..keys].iter().enumerate() {
                        let w = s / sum;
                        let vrow = &v[j * DIM + head * HEAD..][..HEAD];
                        for (o, v) in row.iter_mut().zip(vrow) {
                            *o += w * v;
                        }
                    }
                }
            }
            out
        })
        .collect();
    let mut out = vec![0f32; tokens * DIM];
    for (head, values) in heads.iter().enumerate() {
        for (token, row) in values.chunks_exact(HEAD).enumerate() {
            out[token * DIM + head * HEAD..][..HEAD].copy_from_slice(row);
        }
    }
    out
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(crate) fn read(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    pub(crate) fn db(reference: &[f32], got: &[f32]) -> f64 {
        assert_eq!(reference.len(), got.len());
        let signal: f64 = reference.iter().map(|v| f64::from(*v).powi(2)).sum();
        let noise: f64 = reference
            .iter()
            .zip(got)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        10.0 * (signal / noise).log10()
    }
    /// One captured C++ step: `<dir>/step<N>.{x,timestep,context,out,ref*}.f32` and `.txt`.
    pub(crate) struct Golden {
        pub x: Vec<f32>,
        pub height: usize,
        pub width: usize,
        pub timestep: f32,
        pub context: Vec<f32>,
        pub slots: Vec<i32>,
        pub references: Vec<(Vec<f32>, usize, usize)>,
        pub out: Vec<f32>,
    }
    impl Golden {
        pub(crate) fn load(dir: &Path, step: usize) -> Self {
            let file = |name: &str| dir.join(format!("step{step}.{name}"));
            let text = std::fs::read_to_string(file("txt")).unwrap();
            let (mut width, mut height) = (0, 0);
            let mut slots = Vec::new();
            let mut shapes = Vec::new();
            for line in text.lines() {
                let fields: Vec<&str> = line.split_whitespace().collect();
                match fields.first() {
                    Some(&"x") => {
                        (width, height) = (fields[1].parse().unwrap(), fields[2].parse().unwrap())
                    }
                    Some(&"ref") => {
                        shapes.push((fields[2].parse().unwrap(), fields[1].parse().unwrap()))
                    }
                    Some(&"slots") => {
                        slots = fields[1..].iter().map(|v| v.parse().unwrap()).collect()
                    }
                    _ => {}
                }
            }
            if slots.iter().all(|s| *s == 0) {
                slots.clear();
            }
            let references = shapes
                .iter()
                .enumerate()
                .map(|(i, &(h, w))| (read(&file(&format!("ref{i}.f32"))), h, w))
                .collect();
            Golden {
                x: read(&file("x.f32")),
                height,
                width,
                timestep: read(&file("timestep.f32"))[0],
                context: read(&file("context.f32")),
                slots,
                references,
                out: read(&file("out.f32")),
            }
        }
        pub(crate) fn step(&self) -> Step<'_> {
            Step {
                x: &self.x,
                height: self.height,
                width: self.width,
                timestep: self.timestep,
                context: &self.context,
                slots: &self.slots,
                references: self
                    .references
                    .iter()
                    .map(|(r, h, w)| (&r[..], *h, *w))
                    .collect(),
            }
        }
    }

    #[test]
    fn layout_follows_the_engine() {
        let layout = Layout::new(6, &[0, 0, 1, 0, 0, 0], &[(2, 2), (3, 4)]).unwrap();
        let summary: Vec<_> = layout
            .segments
            .iter()
            .map(|s| (s.start, s.end, s.context_start, s.image))
            .collect();
        assert_eq!(
            summary,
            [
                (0, 2, 0, None),
                (2, 6, 2, Some(0)),
                (6, 9, 3, None),
                (9, 21, 6, Some(1))
            ]
        );
        assert_eq!(layout.prefix, 9);
        assert_eq!(layout.positions[2], [2.0, -1.0, -1.0]);
        assert_eq!(layout.positions[6], [4.0; 3]);
        assert_eq!(layout.positions[9], [7.0, -2.0, -2.0]);
        assert!(Layout::new(4, &[0, 1, 1, 0], &[(2, 2), (2, 2)]).is_err());
    }

    /// `KUNO_QWEN_DIT=<qwen_image_2.1_int8_convrot.safetensors> KUNO_QWEN_GOLDEN=<dir>`:
    /// every captured step against the C++ engine's output.
    #[test]
    #[ignore]
    fn steps_match_cpp_engine() {
        let dit = QwenDit::open(Path::new(&std::env::var("KUNO_QWEN_DIT").unwrap())).unwrap();
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_GOLDEN").unwrap());
        for step in 0.. {
            if !dir.join(format!("step{step}.txt")).exists() {
                assert!(step > 0);
                break;
            }
            let golden = Golden::load(&dir, step);
            let started = std::time::Instant::now();
            let out = dit.forward(&golden.step()).unwrap();
            let db = db(&golden.out, &out);
            println!(
                "step {step}: t {} tokens {}x{} refs {} -> {db:.1} dB ({:.1} s)",
                golden.timestep,
                golden.height,
                golden.width,
                golden.references.len(),
                started.elapsed().as_secs_f64()
            );
            assert!(db > 20.0, "step {step}: {db:.1} dB");
        }
    }
}
