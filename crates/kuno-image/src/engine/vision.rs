//! Qwen3-VL's vision tower as MiniMax H3 uses it: an image becomes rows for the text
//! encoder's language model (one per 32x32 pixels) plus three "deep-stack" sets of rows
//! that the language model adds after its first three layers.
//!
//! A ViT of 27 pre-norm blocks (width 1152, 16 heads of 72, LayerNorm, tanh GELU) over
//! 16x16 patches. Positions enter twice: a learned 48x48 table interpolated bilinearly to the
//! patch grid, and a 2D rotary turn (rows on the first 18 channel pairs of a head, columns on
//! the next 18; pair i is channels i and i + 36). Patches are ordered in 2x2 blocks, so the
//! mergers can take four neighbouring rows as one 4608-wide row: the final merger after a
//! LayerNorm over each patch, the deep-stack mergers (after blocks 8, 16 and 24) with a
//! LayerNorm over the merged row; both then a GELU (erf) between two linear layers.
//!
//! This module holds the layout and the preprocessing, shared with the GPU form
//! (`vision_gpu`), and the exact form in double precision, which arbitrates both.
use std::path::Path;

use rayon::prelude::*;

use super::safetensors::Safetensors;

pub const PATCH: usize = 16;
/// Patches per merged row on each side.
pub const MERGE: usize = 2;
/// Pixels per text encoder row on each side.
pub const FACTOR: usize = PATCH * MERGE;
pub const WIDTH: usize = 1152;
pub const HEADS: usize = 16;
pub const HEAD: usize = WIDTH / HEADS;
pub const HIDDEN: usize = 4304;
pub const LAYERS: usize = 27;
/// Blocks whose output feeds a deep-stack merger, in order.
pub const DEEPSTACK: [usize; 3] = [8, 16, 24];
/// Width of a merged row: four patches.
pub const MERGED: usize = WIDTH * MERGE * MERGE;
/// The text encoder's width.
pub const OUT: usize = 5120;
/// Side of the learned position table.
pub const TABLE: usize = 48;
/// Values in one patch: 3 channels x 16 x 16 (the two time slices see the same image).
pub const PATCH_VALUES: usize = 3 * PATCH * PATCH;
pub const EPS: f64 = 1e-6;
const THETA: f64 = 10000.0;

/// Whether an image of this size can be encoded: whole merged rows, as the engine resizes
/// to, and no more patches than the attention scratch allows.
pub fn supported(width: usize, height: usize) -> bool {
    width.is_multiple_of(FACTOR)
        && height.is_multiple_of(FACTOR)
        && width >= FACTOR
        && height >= FACTOR
        && (width / PATCH) * (height / PATCH) <= 65536
}

/// The patch grid (rows, columns) of an image.
pub fn grid(width: usize, height: usize) -> (usize, usize) {
    (height / PATCH, width / PATCH)
}

/// Each patch's grid position in the model's order: 2x2 blocks row by row, then the
/// patches of a block row by row.
pub fn order(width: usize, height: usize) -> Vec<(usize, usize)> {
    let (gh, gw) = grid(width, height);
    let mut out = Vec::with_capacity(gh * gw);
    for bh in 0..gh / MERGE {
        for bw in 0..gw / MERGE {
            for ih in 0..MERGE {
                for iw in 0..MERGE {
                    out.push((bh * MERGE + ih, bw * MERGE + iw));
                }
            }
        }
    }
    out
}

/// Patch rows `[patches][3 * 16 * 16]` (channel, then pixel row, then column) of an image
/// `[3][height][width]` already scaled to -1..1, in the model's order.
pub fn patches(pixels: &[f32], width: usize, height: usize) -> Vec<f32> {
    let plane = width * height;
    let mut out = Vec::with_capacity(plane * 3);
    for (r, c) in order(width, height) {
        for ch in 0..3 {
            for y in 0..PATCH {
                let at = ch * plane + (r * PATCH + y) * width + c * PATCH;
                out.extend_from_slice(&pixels[at..at + PATCH]);
            }
        }
    }
    out
}

/// The learned position table interpolated to each patch (align-corners bilinear, as the
/// engine and the reference do), `[patches][WIDTH]` in the model's order.
pub fn positions(table: &[f32], width: usize, height: usize) -> Vec<f64> {
    let (gh, gw) = grid(width, height);
    let last = (TABLE - 1) as f64;
    let at = |i: usize, n: usize| -> (usize, usize, f64) {
        let p = if n == 1 {
            0.0
        } else {
            last * i as f64 / (n - 1) as f64
        };
        let lo = p.floor() as usize;
        (lo, (lo + 1).min(TABLE - 1), p - lo as f64)
    };
    let mut out = Vec::with_capacity(gh * gw * WIDTH);
    for (r, c) in order(width, height) {
        let (h0, h1, dh) = at(r, gh);
        let (w0, w1, dw) = at(c, gw);
        // The engine rounds each corner's weight to f32.
        let corners = [
            (h0 * TABLE + w0, ((1.0 - dh) * (1.0 - dw)) as f32 as f64),
            (h0 * TABLE + w1, ((1.0 - dh) * dw) as f32 as f64),
            (h1 * TABLE + w0, (dh * (1.0 - dw)) as f32 as f64),
            (h1 * TABLE + w1, (dh * dw) as f32 as f64),
        ];
        for ch in 0..WIDTH {
            out.push(
                corners
                    .iter()
                    .map(|&(row, weight)| table[row * WIDTH + ch] as f64 * weight)
                    .sum(),
            );
        }
    }
    out
}

/// Each patch's rotary angles `[patches][HEAD / 2]`: pair i < 18 turns by row * f_i, pair
/// 18 + i by column * f_i, with f_i = 10000^(-2i / 36).
pub fn angles(width: usize, height: usize) -> Vec<f64> {
    let quarter = HEAD / 4;
    let mut out = Vec::with_capacity((width / PATCH) * (height / PATCH) * HEAD / 2);
    for (r, c) in order(width, height) {
        for p in [r, c] {
            for i in 0..quarter {
                // The engine computes the frequency in f32.
                let scale = (2 * i) as f32 / (2 * quarter) as f32;
                let frequency = 1.0 / (THETA as f32).powf(scale);
                out.push(p as f64 * frequency as f64);
            }
        }
    }
    out
}

pub struct Vision {
    pub(super) file: Safetensors,
}
impl Vision {
    /// The vision tower inside the text encoder's file.
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        if file.count("visual.blocks.") != LAYERS
            || file.count("visual.deepstack_merger_list.") != DEEPSTACK.len()
        {
            return Err("model_file_invalid");
        }
        Ok(Self { file })
    }
    pub(super) fn tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, &'static str> {
        self.file.floats(&format!("visual.{name}"), shape)
    }
    /// The patch embedding as a `[WIDTH][PATCH_VALUES]` matrix: the model's kernel spans two
    /// time slices of one image, so they are summed.
    pub fn patch_matrix(&self) -> Result<Vec<f32>, &'static str> {
        let w = self.tensor("patch_embed.proj.weight", &[WIDTH, 3, 2, PATCH, PATCH])?;
        let area = PATCH * PATCH;
        let mut out = vec![0f32; WIDTH * PATCH_VALUES];
        for o in 0..WIDTH {
            for c in 0..3 {
                for i in 0..area {
                    let first = w[((o * 3 + c) * 2) * area + i] as f64;
                    let second = w[((o * 3 + c) * 2 + 1) * area + i] as f64;
                    out[o * PATCH_VALUES + c * area + i] = (first + second) as f32;
                }
            }
        }
        Ok(out)
    }
    /// The image's rows for the text encoder `[tokens][OUT]`, then the three deep-stack sets,
    /// from an image `[3][height][width]` scaled to -1..1.
    pub fn encode_exact(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<[Vec<f64>; 4], &'static str> {
        if !supported(width, height) || pixels.len() != 3 * width * height {
            return Err("invalid_input");
        }
        let n = (width / PATCH) * (height / PATCH);
        let patch = patches(pixels, width, height);
        let matrix = self.patch_matrix()?;
        let bias = self.tensor("patch_embed.proj.bias", &[WIDTH])?;
        let table = self.tensor("pos_embed.weight", &[TABLE * TABLE, WIDTH])?;
        let mut x = positions(&table, width, height);
        let patch64: Vec<f64> = patch.iter().map(|&v| v as f64).collect();
        let embedded = linear(&patch64, PATCH_VALUES, &matrix, &bias);
        x.iter_mut().zip(&embedded).for_each(|(x, e)| *x += e);
        let turns = angles(width, height);
        let mut deep = Vec::new();
        for layer in 0..LAYERS {
            self.block(layer, &mut x, n, &turns)?;
            if let Some(i) = DEEPSTACK.iter().position(|&l| l == layer) {
                // Four neighbouring patches as one row, normed as a whole.
                let name = format!("deepstack_merger_list.{i}");
                let h = layer_norm(&x, MERGED, &self.norm(&name, MERGED)?);
                deep.push(self.merger_mlp(&name, &h)?);
            }
        }
        let h = layer_norm(&x, WIDTH, &self.norm("merger", WIDTH)?);
        let main = self.merger_mlp("merger", &h)?;
        let [a, b, c]: [Vec<f64>; 3] = deep.try_into().map_err(|_| "model_file_invalid")?;
        Ok([main, a, b, c])
    }
    fn norm(&self, name: &str, width: usize) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
        Ok((
            self.tensor(&format!("{name}.norm.weight"), &[width])?,
            self.tensor(&format!("{name}.norm.bias"), &[width])?,
        ))
    }
    fn dense(&self, name: &str, rows: usize, columns: usize) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
        Ok((
            self.tensor(&format!("{name}.weight"), &[rows, columns])?,
            self.tensor(&format!("{name}.bias"), &[rows])?,
        ))
    }
    fn merger_mlp(&self, name: &str, h: &[f64]) -> Result<Vec<f64>, &'static str> {
        let (w1, b1) = self.dense(&format!("{name}.linear_fc1"), MERGED, MERGED)?;
        let (w2, b2) = self.dense(&format!("{name}.linear_fc2"), OUT, MERGED)?;
        let mut g = linear(h, MERGED, &w1, &b1);
        g.iter_mut().for_each(|v| *v = gelu_erf(*v));
        Ok(linear(&g, MERGED, &w2, &b2))
    }
    fn block(&self, layer: usize, x: &mut [f64], n: usize, turns: &[f64]) -> Result<(), &'static str> {
        let p = format!("blocks.{layer}");
        let ln = |which: &str| -> Result<(Vec<f32>, Vec<f32>), &'static str> {
            Ok((
                self.tensor(&format!("{p}.{which}.weight"), &[WIDTH])?,
                self.tensor(&format!("{p}.{which}.bias"), &[WIDTH])?,
            ))
        };
        let h = layer_norm(x, WIDTH, &ln("norm1")?);
        let (wqkv, bqkv) = self.dense(&format!("{p}.attn.qkv"), 3 * WIDTH, WIDTH)?;
        let qkv = linear(&h, WIDTH, &wqkv, &bqkv);
        let half = HEAD / 2;
        // q, k, v as [head][token][HEAD], queries and keys turned.
        let split = |kind: usize, turn: bool| -> Vec<f64> {
            let mut out = vec![0f64; HEADS * n * HEAD];
            for t in 0..n {
                for hd in 0..HEADS {
                    let from = &qkv[t * 3 * WIDTH + kind * WIDTH + hd * HEAD..][..HEAD];
                    let to = &mut out[(hd * n + t) * HEAD..][..HEAD];
                    if !turn {
                        to.copy_from_slice(from);
                        continue;
                    }
                    for i in 0..half {
                        let (s, c) = turns[t * half + i].sin_cos();
                        to[i] = from[i] * c - from[i + half] * s;
                        to[i + half] = from[i] * s + from[i + half] * c;
                    }
                }
            }
            out
        };
        let (q, k, v) = (split(0, true), split(1, true), split(2, false));
        let scale = 1.0 / (HEAD as f64).sqrt();
        let mut attended = vec![0f64; n * WIDTH];
        attended
            .par_chunks_mut(WIDTH)
            .enumerate()
            .for_each(|(t, row)| {
                let mut scores = vec![0f64; n];
                for hd in 0..HEADS {
                    let qr = &q[(hd * n + t) * HEAD..][..HEAD];
                    let mut top = f64::NEG_INFINITY;
                    for (s, score) in scores.iter_mut().enumerate() {
                        let kr = &k[(hd * n + s) * HEAD..][..HEAD];
                        *score = qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f64>() * scale;
                        top = top.max(*score);
                    }
                    let mut total = 0.0;
                    scores.iter_mut().for_each(|s| {
                        *s = (*s - top).exp();
                        total += *s;
                    });
                    let out = &mut row[hd * HEAD..][..HEAD];
                    for (s, &w) in scores.iter().enumerate() {
                        let vr = &v[(hd * n + s) * HEAD..][..HEAD];
                        out.iter_mut().zip(vr).for_each(|(o, v)| *o += w / total * v);
                    }
                }
            });
        let (wp, bp) = self.dense(&format!("{p}.attn.proj"), WIDTH, WIDTH)?;
        let projected = linear(&attended, WIDTH, &wp, &bp);
        x.iter_mut().zip(&projected).for_each(|(x, d)| *x += d);
        let h = layer_norm(x, WIDTH, &ln("norm2")?);
        let (w1, b1) = self.dense(&format!("{p}.mlp.linear_fc1"), HIDDEN, WIDTH)?;
        let (w2, b2) = self.dense(&format!("{p}.mlp.linear_fc2"), WIDTH, HIDDEN)?;
        let mut g = linear(&h, WIDTH, &w1, &b1);
        g.iter_mut().for_each(|v| *v = gelu_tanh(*v));
        let down = linear(&g, HIDDEN, &w2, &b2);
        x.iter_mut().zip(&down).for_each(|(x, d)| *x += d);
        Ok(())
    }
}

/// `x [n][columns] * w^T + b` for `w [rows][columns]`.
fn linear(x: &[f64], columns: usize, w: &[f32], b: &[f32]) -> Vec<f64> {
    let rows = b.len();
    let n = x.len() / columns;
    let mut out = vec![0f64; n * rows];
    out.par_chunks_mut(rows).enumerate().for_each(|(t, row)| {
        let xr = &x[t * columns..][..columns];
        for (r, o) in row.iter_mut().enumerate() {
            let wr = &w[r * columns..][..columns];
            *o = b[r] as f64 + xr.iter().zip(wr).map(|(a, &b)| a * b as f64).sum::<f64>();
        }
    });
    out
}
/// LayerNorm over rows of `width` with weight and bias.
fn layer_norm(x: &[f64], width: usize, (weight, bias): &(Vec<f32>, Vec<f32>)) -> Vec<f64> {
    let mut out = vec![0f64; x.len()];
    for (row, to) in x.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        let mean = row.iter().sum::<f64>() / width as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / width as f64;
        let inv = 1.0 / (var + EPS).sqrt();
        for i in 0..width {
            to[i] = (row[i] - mean) * inv * weight[i] as f64 + bias[i] as f64;
        }
    }
    out
}
pub(super) fn gelu_tanh(x: f64) -> f64 {
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2 / pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}
pub(super) fn gelu_erf(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}
/// The error function to about 1e-16 (Abramowitz-Stegun 7.1.26 is too coarse here): a
/// Taylor series near zero and a continued fraction for the complement beyond.
fn erf(x: f64) -> f64 {
    let a = x.abs();
    let value = if a < 2.5 {
        // erf(x) = 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1))
        let mut term = a;
        let mut sum = a;
        let x2 = a * a;
        let mut n = 0.0;
        loop {
            n += 1.0;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else {
        // erfc(a) = exp(-a^2)/sqrt(pi) * 1/(a + 1/2/(a + 1/(a + 3/2/(a + ...))))
        let mut f = 0.0;
        for k in (1..60).rev() {
            f = (k as f64 / 2.0) / (a + f);
        }
        1.0 - (-a * a).exp() / std::f64::consts::PI.sqrt() / (a + f)
    };
    value.copysign(x)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn layout_and_helpers() {
        assert_eq!(HEAD, 72);
        assert_eq!(MERGED, 4608);
        let order = order(64, 32);
        // 4 x 2 patches: two 2x2 blocks side by side.
        assert_eq!(&order[..6], &[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3)]);
        assert!(supported(832, 544) && !supported(840, 544));
        for (x, want) in [(0.5, 0.520_499_877_813_046_5), (1.0, 0.842_700_792_949_714_9), (3.0, 0.999_977_909_503_001_4)] {
            assert!((erf(x) - want).abs() < 1e-14, "{x}");
        }
        assert!((gelu_erf(-1.0) + 0.158_655_253_931_457_05).abs() < 1e-14);
        let a = angles(64, 32);
        // Patch (0, 1): column 1 on pairs 18.., row 0 on pairs ..18.
        assert_eq!(a[36], 0.0);
        assert_eq!(a[36 + 18], 1.0);
    }

    /// A smooth synthetic image scaled to -1..1.
    pub fn image(width: usize, height: usize, seed: u64) -> Vec<f32> {
        crate::engine::video_encoder::tests::frame(width, height, seed)
            .into_iter()
            .map(|v| v * 2.0 - 1.0)
            .collect()
    }
}
