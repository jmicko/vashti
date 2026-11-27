//! Weights and the small set of operations the LFM3-Audio models need, on the CPU: linear
//! layers over f32 rows (widened F16/F32 weights) or packed Q8_0 rows (the backbone), norms,
//! activations and NEOX rotary embedding.
use rayon::prelude::*;

use kuno_text::{GgufFile, TensorType};

use crate::Result;

/// A weight matrix: `rows` outputs over `cols` inputs.
pub enum Linear {
    F32 {
        w: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    /// GGML Q8_0 rows: per 32 values an f16 scale and 32 signed bytes.
    Q8 {
        w: Vec<u8>,
        rows: usize,
        cols: usize,
    },
    /// Half-precision rows, widened as they are read (half the memory traffic of f32).
    F16 {
        w: Vec<u16>,
        rows: usize,
        cols: usize,
    },
}

impl Linear {
    pub fn load(g: &GgufFile, name: &str) -> Result<Self> {
        let view = g.tensor(name).ok_or("model_tensor_missing")?;
        let info = view.info();
        let (rows, cols) = (info.row_count() as usize, info.row_len() as usize);
        let mut bytes = vec![0u8; info.byte_len() as usize];
        view.read_range(0, &mut bytes)
            .map_err(|_| "model_read_failed")?;
        if info.dtype() == TensorType::Q8_0 {
            return Ok(Self::Q8 {
                w: bytes,
                rows,
                cols,
            });
        }
        if info.dtype() == TensorType::F16 {
            let w = bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            return Ok(Self::F16 { w, rows, cols });
        }
        let mut w = vec![0f32; rows * cols];
        let rb = info.row_bytes() as usize;
        w.par_chunks_exact_mut(cols)
            .zip(bytes.par_chunks_exact(rb))
            .try_for_each(|(out, packed)| info.dtype().decode(packed, out))
            .map_err(|_| "model_decode_failed")?;
        Ok(Self::F32 { w, rows, cols })
    }
    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } | Self::Q8 { rows, .. } | Self::F16 { rows, .. } => *rows,
        }
    }
    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } | Self::Q8 { cols, .. } | Self::F16 { cols, .. } => *cols,
        }
    }
    /// Bytes the weights take.
    pub fn bytes(&self) -> usize {
        match self {
            Self::F32 { w, .. } => w.len() * 4,
            Self::Q8 { w, .. } => w.len(),
            Self::F16 { w, .. } => w.len() * 2,
        }
    }
    /// The values of an f32 matrix (row-major), or `None` for packed rows.
    pub fn f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32 { w, .. } => Some(w),
            Self::Q8 { .. } | Self::F16 { .. } => None,
        }
    }
    /// Row `r` decoded to f32 (embedding lookups).
    pub fn row(&self, r: usize) -> Vec<f32> {
        match self {
            Self::F32 { w, cols, .. } => w[r * cols..][..*cols].to_vec(),
            Self::F16 { w, cols, .. } => w[r * cols..][..*cols].iter().map(|b| f16(*b)).collect(),
            Self::Q8 { w, cols, .. } => {
                let rb = cols / 32 * 34;
                let packed = &w[r * rb..][..rb];
                let mut out = vec![0f32; *cols];
                for (b, block) in packed.chunks_exact(34).enumerate() {
                    let d = f16(u16::from_le_bytes([block[0], block[1]]));
                    for i in 0..32 {
                        out[b * 32 + i] = d * f32::from(block[2 + i] as i8);
                    }
                }
                out
            }
        }
    }
    /// `x` is `n` rows of `cols`; returns `n` rows of `rows` (each output row a dot product).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let (rows, cols) = (self.rows(), self.cols());
        let n = x.len() / cols;
        assert_eq!(n * cols, x.len());
        // Per output row (decoded once), its dot product with every input row.
        let mut out_t = vec![0f32; rows * n];
        match self {
            Self::F32 { w, .. } => {
                out_t
                    .par_chunks_mut(n)
                    .with_min_len(16)
                    .enumerate()
                    .for_each(|(r, o)| {
                        let wr = &w[r * cols..][..cols];
                        for (t, o) in o.iter_mut().enumerate() {
                            *o = dot(wr, &x[t * cols..][..cols]);
                        }
                    });
            }
            Self::F16 { w, .. } => {
                // Rows in batches (small matrices otherwise cost more in scheduling than in
                // arithmetic); each row's halves widened once for four input rows at a time.
                out_t
                    .par_chunks_mut(n)
                    .with_min_len(16)
                    .enumerate()
                    .for_each(|(r, o)| {
                        let wr = &w[r * cols..][..cols];
                        let mut t = 0;
                        while t + 4 <= n {
                            let xs = [0, 1, 2, 3].map(|i| &x[(t + i) * cols..][..cols]);
                            o[t..t + 4].copy_from_slice(&dot_f16_x4(wr, xs));
                            t += 4;
                        }
                        for t in t..n {
                            o[t] = dot_f16(wr, &x[t * cols..][..cols]);
                        }
                    });
            }
            Self::Q8 { w, .. } => {
                let rb = cols / 32 * 34;
                out_t
                    .par_chunks_mut(n)
                    .with_min_len(16)
                    .enumerate()
                    .for_each(|(r, o)| {
                        let packed = &w[r * rb..][..rb];
                        for (t, o) in o.iter_mut().enumerate() {
                            *o = dot_q8(packed, &x[t * cols..][..cols]);
                        }
                    });
            }
        }
        if n == 1 {
            return out_t;
        }
        let mut out = vec![0f32; n * rows];
        for r in 0..rows {
            for t in 0..n {
                out[t * rows + r] = out_t[r * n + t];
            }
        }
        out
    }
    /// `forward` plus a bias per output.
    pub fn forward_bias(&self, x: &[f32], bias: &[f32]) -> Vec<f32> {
        let mut y = self.forward(x);
        for row in y.chunks_exact_mut(bias.len()) {
            for (v, b) in row.iter_mut().zip(bias) {
                *v += b;
            }
        }
        y
    }
}

pub fn f16(bits: u16) -> f32 {
    kuno_text::quant::f16_to_f32(bits)
}

fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: the CPU has AVX2 and FMA.
        return unsafe { x86::dot(a, b) };
    }
    dot_plain(a, b)
}

fn dot_plain(a: &[f32], b: &[f32]) -> f32 {
    // Eight partial sums vectorize; their order is fixed, so results are deterministic.
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// Row `r` of a matrix dotted with `x` (the depth decoder's per-codebook logits).
pub fn dot_row(m: &Linear, r: usize, x: &[f32]) -> f32 {
    match m {
        Linear::F32 { w, cols, .. } => dot(&w[r * cols..][..*cols], x),
        Linear::F16 { w, cols, .. } => dot_f16(&w[r * cols..][..*cols], x),
        Linear::Q8 { w, cols, .. } => {
            let rb = cols / 32 * 34;
            dot_q8(&w[r * rb..][..rb], x)
        }
    }
}

/// One half-precision row against four f32 rows.
fn dot_f16_x4(w: &[u16], x: [&[f32]; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if avx2() && std::is_x86_feature_detected!("f16c") {
        // SAFETY: the CPU has AVX2, FMA and F16C.
        return unsafe { x86::dot_f16_x4(w, x) };
    }
    x.map(|x| dot_f16(w, x))
}

fn dot_f16(w: &[u16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2() && std::is_x86_feature_detected!("f16c") {
        // SAFETY: the CPU has AVX2, FMA and F16C.
        return unsafe { x86::dot_f16(w, x) };
    }
    w.iter().zip(x).map(|(a, b)| f16(*a) * b).sum()
}

fn dot_q8(packed: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: the CPU has AVX2 and FMA.
        return unsafe { x86::dot_q8(packed, x) };
    }
    let mut total = 0f32;
    for (block, xs) in packed.chunks_exact(34).zip(x.chunks_exact(32)) {
        let d = f16(u16::from_le_bytes([block[0], block[1]]));
        let mut acc = [0f32; 8];
        for (q, xv) in block[2..].chunks_exact(8).zip(xs.chunks_exact(8)) {
            for i in 0..8 {
                acc[i] += f32::from(q[i] as i8) * xv[i];
            }
        }
        total += d * acc.iter().sum::<f32>();
    }
    total
}

/// A small f32 tensor read whole (norm weights, biases, conv taps).
pub fn vector(g: &GgufFile, name: &str) -> Result<Vec<f32>> {
    match Linear::load(g, name)? {
        Linear::F32 { w, .. } => Ok(w),
        q => Ok((0..q.rows()).flat_map(|r| q.row(r)).collect()),
    }
}

/// RMS norm of each `width` row, times `weight` (one per value, or one per row group).
pub fn rms_norm(x: &mut [f32], weight: Option<&[f32]>, width: usize, eps: f32) {
    for row in x.chunks_exact_mut(width) {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let s = 1.0 / (mean + eps).sqrt();
        match weight {
            Some(w) => row.iter_mut().zip(w).for_each(|(v, w)| *v = *v * s * w),
            None => row.iter_mut().for_each(|v| *v *= s),
        }
    }
}
pub fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let width = weight.len();
    for row in x.chunks_exact_mut(width) {
        let mean = row.iter().sum::<f32>() / width as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
        let s = 1.0 / (var + eps).sqrt();
        for ((v, w), b) in row.iter_mut().zip(weight).zip(bias) {
            *v = (*v - mean) * s * w + b;
        }
    }
}
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
/// GELU with the error function (ggml's `gelu_erf`).
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}
fn erf(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26 is too coarse; use the double-precision series/continued form.
    let x = f64::from(x);
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

/// NEOX rotary embedding of one head (`d` values) at position `pos`: pairs (i, i + d/2) turned
/// by `pos * base^(-2i/d) / factor_i`.
pub fn rope_neox(head: &mut [f32], pos: usize, base: f32, factors: Option<&[f32]>) {
    let d = head.len();
    let half = d / 2;
    for i in 0..half {
        let mut theta = pos as f32 * base.powf(-2.0 * i as f32 / d as f32);
        if let Some(f) = factors {
            theta /= f[i];
        }
        let (s, c) = theta.sin_cos();
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * c - b * s;
        head[i + half] = a * s + b * c;
    }
}

/// Softmax over `scores`, in place.
pub fn softmax(scores: &mut [f32]) {
    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - top).exp();
        sum += *s;
    }
    scores.iter_mut().for_each(|s| *s /= sum);
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    unsafe fn sum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// Four eight-wide FMA accumulators.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (j, s) in acc.iter_mut().enumerate() {
                *s = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 8 * j)),
                    _mm256_loadu_ps(pb.add(i + 8 * j)),
                    *s,
                );
            }
            i += 32;
        }
        while i + 8 <= n {
            acc[0] = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i)),
                _mm256_loadu_ps(pb.add(i)),
                acc[0],
            );
            i += 8;
        }
        let mut total = sum(_mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]),
        ));
        while i < n {
            total += a[i] * b[i];
            i += 1;
        }
        total
    }

    /// One half-precision row widened once per eight values for four f32 rows.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16_x4(w: &[u16], x: [&[f32]; 4]) -> [f32; 4] {
        let n = x.iter().fold(w.len(), |n, x| n.min(x.len()));
        let pw = w.as_ptr();
        let px = x.map(|x| x.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 8 <= n {
            let h = _mm256_cvtph_ps(_mm_loadu_si128(pw.add(i) as *const __m128i));
            for j in 0..4 {
                acc[j] = _mm256_fmadd_ps(h, _mm256_loadu_ps(px[j].add(i)), acc[j]);
            }
            i += 8;
        }
        let mut out = [0f32; 4];
        for j in 0..4 {
            out[j] = sum(acc[j]);
            for k in i..n {
                out[j] += super::f16(w[k]) * x[j][k];
            }
        }
        out
    }

    /// Half-precision values widened eight at a time (F16C) against f32 values.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len().min(x.len());
        let (pw, px) = (w.as_ptr(), x.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (j, s) in acc.iter_mut().enumerate() {
                let h = _mm256_cvtph_ps(_mm_loadu_si128(pw.add(i + 8 * j) as *const __m128i));
                *s = _mm256_fmadd_ps(h, _mm256_loadu_ps(px.add(i + 8 * j)), *s);
            }
            i += 32;
        }
        while i + 8 <= n {
            let h = _mm256_cvtph_ps(_mm_loadu_si128(pw.add(i) as *const __m128i));
            acc[0] = _mm256_fmadd_ps(h, _mm256_loadu_ps(px.add(i)), acc[0]);
            i += 8;
        }
        let mut total = sum(_mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]),
        ));
        while i < n {
            total += super::f16(w[i]) * x[i];
            i += 1;
        }
        total
    }

    /// Q8_0 rows against f32 values: per block the bytes widened to f32 and multiplied in.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8(packed: &[u8], x: &[f32]) -> f32 {
        let mut total = _mm256_setzero_ps();
        for (block, xs) in packed.chunks_exact(34).zip(x.chunks_exact(32)) {
            let d = _mm256_set1_ps(super::f16(u16::from_le_bytes([block[0], block[1]])));
            let q = block.as_ptr().add(2);
            let px = xs.as_ptr();
            let mut acc = _mm256_setzero_ps();
            for j in 0..4 {
                let bytes = _mm_loadl_epi64(q.add(8 * j) as *const __m128i);
                let w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
                acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(px.add(8 * j)), acc);
            }
            total = _mm256_fmadd_ps(acc, d, total);
        }
        sum(total)
    }
}
