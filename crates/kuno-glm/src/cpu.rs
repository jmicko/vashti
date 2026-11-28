//! The CPU's share of the routed experts: matrix-vector products over the packed GGUF rows
//! with the input quantized to 8 bits per 256 values (GGML's Q8_K), integer dot products in
//! AVX2. These follow GGML's x86 kernels (`ggml_vec_dot_iq2_s_q8_K`, `iq3_s`, `iq4_xs`; MIT),
//! so their sums match llama.cpp's CPU path. Experts are IQ2_S (gate, up) and IQ3_S (down),
//! with IQ4_XS for three down projections and IQ3_S gate/up in block 11.
//!
//! Rows are in the device's layout (`experts`): for IQ2_S and IQ3_S each block's body
//! (80 or 108 bytes, the GGUF block without its leading f16 scale) in `row`, the scales
//! apart in `ds`; IQ4_XS blocks whole (136 bytes, `ds` empty).
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use kuno_text::TensorType;
use kuno_text::quant::{IQ2_S_GRID, IQ3_S_GRID, IQ4_NL_VALUES};

use crate::Result;

const QK: usize = 256;

/// An input quantized per 256 values: `x[i] ~ d[i / 256] * q[i]`.
pub struct Q8K {
    pub d: Vec<f32>,
    pub q: Vec<i8>,
}

impl Q8K {
    /// GGML's `quantize_row_q8_K_ref`: the scale maps the largest magnitude to -127 or 127.
    pub fn new(x: &[f32]) -> Self {
        assert!(x.len().is_multiple_of(QK));
        let mut d = Vec::with_capacity(x.len() / QK);
        let mut q = vec![0i8; x.len()];
        for (block, out) in x.chunks_exact(QK).zip(q.chunks_exact_mut(QK)) {
            let (mut amax, mut max) = (0f32, 0f32);
            for &v in block {
                if v.abs() > amax {
                    amax = v.abs();
                    max = v;
                }
            }
            if amax == 0.0 {
                d.push(0.0);
                continue;
            }
            let iscale = -127.0 / max;
            for (o, &v) in out.iter_mut().zip(block) {
                *o = ((iscale * v).round_ties_even() as i32).min(127) as i8;
            }
            d.push(1.0 / iscale);
        }
        Self { d, q }
    }
}

fn half(b: &[u8]) -> f32 {
    kuno_text::quant::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

/// Bytes per 256 values in the device's layout (the body; scales apart for IQ2_S/IQ3_S).
pub fn block_bytes(kind: TensorType) -> Option<usize> {
    match kind {
        TensorType::Iq2S => Some(80),
        TensorType::Iq3S => Some(108),
        TensorType::Iq4Xs => Some(136),
        _ => None,
    }
}

pub fn supported(kind: TensorType) -> bool {
    block_bytes(kind).is_some()
}

/// One row (`row` packed bytes) dotted with the quantized input.
pub fn dot(kind: TensorType, row: &[u8], ds: &[u8], x: &Q8K) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 and FMA were detected; the kernels read only within `row` and `x`,
        // whose lengths are checked against each other.
        unsafe {
            return match kind {
                TensorType::Iq2S => dot_iq2_s(row, ds, x),
                TensorType::Iq3S => dot_iq3_s(row, ds, x),
                TensorType::Iq4Xs => dot_iq4_xs(row, x),
                _ => panic!("unsupported expert type"),
            };
        }
    }
    dot_scalar(kind, row, ds, x)
}

/// `out[r] = row r . x` over `rows` (bodies `row_bytes` long; scales `2 * blocks` per row).
pub fn matvec(
    kind: TensorType,
    rows: &[u8],
    ds: &[u8],
    row_bytes: usize,
    x: &Q8K,
    out: &mut [f32],
) {
    let blocks = x.q.len() / QK;
    for (r, (o, row)) in out.iter_mut().zip(rows.chunks_exact(row_bytes)).enumerate() {
        let d = if ds.is_empty() {
            &[][..]
        } else {
            &ds[r * 2 * blocks..][..2 * blocks]
        };
        *o = dot(kind, row, d, x);
    }
}

/// The same sums without SIMD, from the decoded integer codes (tests and other CPUs).
pub fn dot_scalar(kind: TensorType, row: &[u8], ds: &[u8], x: &Q8K) -> f32 {
    let bb = block_bytes(kind).expect("unsupported expert type");
    assert_eq!(row.len() / bb * QK, x.q.len());
    let mut total = 0f32;
    for (i, b) in row.chunks_exact(bb).enumerate() {
        let q8 = &x.q[i * QK..][..QK];
        let d = if kind == TensorType::Iq4Xs {
            half(b)
        } else {
            half(&ds[i * 2..])
        } * x.d[i];
        let mut sum = 0i32;
        match kind {
            TensorType::Iq2S => {
                let (qs, signs, qh, scales) = (&b[0..32], &b[32..64], &b[64..72], &b[72..80]);
                for ib in 0..8 {
                    let mut part = [0i32; 2];
                    for l in 0..4 {
                        let index = usize::from(qs[ib * 4 + l])
                            | ((usize::from(qh[ib]) << (8 - 2 * l)) & 0x300);
                        let grid = IQ2_S_GRID[index].to_le_bytes();
                        let s = signs[ib * 4 + l];
                        for j in 0..8 {
                            let v = i32::from(grid[j]) * if s >> j & 1 == 1 { -1 } else { 1 };
                            part[l / 2] += v * i32::from(q8[ib * 32 + l * 8 + j]);
                        }
                    }
                    let lo = i32::from(scales[ib] & 15) * 2 + 1;
                    let hi = i32::from(scales[ib] >> 4) * 2 + 1;
                    sum += part[0] * lo + part[1] * hi;
                }
                total += d * sum as f32 * 0.125;
                continue;
            }
            TensorType::Iq3S => {
                let (qs, qh, signs, scales) = (&b[0..64], &b[64..72], &b[72..104], &b[104..108]);
                for ib in 0..8 {
                    let mut part = 0i32;
                    for l in 0..8 {
                        let index =
                            usize::from(qs[ib * 8 + l]) | ((usize::from(qh[ib]) << (8 - l)) & 256);
                        let grid = IQ3_S_GRID[index].to_le_bytes();
                        let s = signs[ib * 4 + l / 2];
                        for j in 0..4 {
                            let bit = (l % 2) * 4 + j;
                            let v = i32::from(grid[j]) * if s >> bit & 1 == 1 { -1 } else { 1 };
                            part += v * i32::from(q8[ib * 32 + l * 4 + j]);
                        }
                    }
                    let ls = if ib % 2 == 0 {
                        scales[ib / 2] & 15
                    } else {
                        scales[ib / 2] >> 4
                    };
                    sum += part * (2 * i32::from(ls) + 1);
                }
            }
            TensorType::Iq4Xs => {
                let sh = u16::from_le_bytes([b[2], b[3]]);
                let (sl, qs) = (&b[4..8], &b[8..136]);
                for ib in 0..8 {
                    let low = if ib % 2 == 0 {
                        sl[ib / 2] & 15
                    } else {
                        sl[ib / 2] >> 4
                    };
                    let ls = (i32::from(low) | ((i32::from(sh) >> (2 * ib) & 3) << 4)) - 32;
                    let mut part = 0i32;
                    for j in 0..16 {
                        let byte = qs[ib * 16 + j];
                        part += i32::from(IQ4_NL_VALUES[usize::from(byte & 15)])
                            * i32::from(q8[ib * 32 + j]);
                        part += i32::from(IQ4_NL_VALUES[usize::from(byte >> 4)])
                            * i32::from(q8[ib * 32 + 16 + j]);
                    }
                    sum += part * ls;
                }
            }
            _ => unreachable!(),
        }
        total += d * sum as f32;
    }
    total
}

#[cfg(target_arch = "x86_64")]
const SIGN_SHUFFLE: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
];
#[cfg(target_arch = "x86_64")]
const SIGN_BITS: [u8; 32] = [
    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4,
    8, 16, 32, 64, 128,
];

/// `q8` with the sign of each byte flipped where the 32 bits of `signs` say.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_signs(signs: u32, q8: __m256i, shuffle: __m256i, bits: __m256i) -> __m256i {
    let s = _mm256_and_si256(
        _mm256_shuffle_epi8(_mm256_set1_epi32(signs as i32), shuffle),
        bits,
    );
    let s = _mm256_cmpeq_epi8(s, bits);
    _mm256_sub_epi8(_mm256_xor_si256(s, q8), s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq2_s(row: &[u8], ds: &[u8], x: &Q8K) -> f32 {
    let blocks = row.len() / 80;
    assert_eq!(blocks * QK, x.q.len());
    let shuffle = unsafe { _mm256_loadu_si256(SIGN_SHUFFLE.as_ptr().cast()) };
    let bits = unsafe { _mm256_loadu_si256(SIGN_BITS.as_ptr().cast()) };
    let mut acc = _mm256_setzero_ps();
    for i in 0..blocks {
        let b = &row[i * 80..][..80];
        let d = half(&ds[i * 2..]) * x.d[i];
        let (qs, signs, qh, scales) = (&b[0..32], &b[32..64], &b[64..72], &b[72..80]);
        let q8 = &x.q[i * QK..][..QK];
        let mut sums = _mm256_setzero_si256();
        for ib in (0..8).step_by(2) {
            // SAFETY: each load is 32 bytes within this block's 256 quantized values.
            let (q8a, q8b) = unsafe {
                (
                    _mm256_loadu_si256(q8[ib * 32..].as_ptr().cast()),
                    _mm256_loadu_si256(q8[ib * 32 + 32..].as_ptr().cast()),
                )
            };
            let g = |k: usize, h: usize, shift: usize| {
                IQ2_S_GRID[usize::from(qs[k]) | ((usize::from(qh[h]) << shift) & 0x300)] as i64
            };
            let a = _mm256_set_epi64x(
                g(ib * 4 + 3, ib, 2),
                g(ib * 4 + 2, ib, 4),
                g(ib * 4 + 1, ib, 6),
                g(ib * 4, ib, 8),
            );
            let c = _mm256_set_epi64x(
                g(ib * 4 + 7, ib + 1, 2),
                g(ib * 4 + 6, ib + 1, 4),
                g(ib * 4 + 5, ib + 1, 6),
                g(ib * 4 + 4, ib + 1, 8),
            );
            let s =
                |k: usize| u32::from_le_bytes([signs[k], signs[k + 1], signs[k + 2], signs[k + 3]]);
            let xa = unsafe { apply_signs(s(ib * 4), q8a, shuffle, bits) };
            let xc = unsafe { apply_signs(s(ib * 4 + 4), q8b, shuffle, bits) };
            let da = _mm256_maddubs_epi16(a, xa);
            let dc = _mm256_maddubs_epi16(c, xc);
            // Per 16 values one scale: 2 * nibble + 1 (low nibble first).
            let sc = |v: u8| 2 * i16::from(v) + 1;
            let (s0, s1, s2, s3) = (
                sc(scales[ib] & 15),
                sc(scales[ib] >> 4),
                sc(scales[ib + 1] & 15),
                sc(scales[ib + 1] >> 4),
            );
            let ma = _mm256_set_epi16(
                s1, s1, s1, s1, s1, s1, s1, s1, s0, s0, s0, s0, s0, s0, s0, s0,
            );
            let mc = _mm256_set_epi16(
                s3, s3, s3, s3, s3, s3, s3, s3, s2, s2, s2, s2, s2, s2, s2, s2,
            );
            sums = _mm256_add_epi32(
                sums,
                _mm256_add_epi32(_mm256_madd_epi16(da, ma), _mm256_madd_epi16(dc, mc)),
            );
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sums), acc);
    }
    0.125 * unsafe { hsum(acc) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq3_s(row: &[u8], ds: &[u8], x: &Q8K) -> f32 {
    let blocks = row.len() / 108;
    assert_eq!(blocks * QK, x.q.len());
    let shuffle = unsafe { _mm256_loadu_si256(SIGN_SHUFFLE.as_ptr().cast()) };
    let bits = unsafe { _mm256_loadu_si256(SIGN_BITS.as_ptr().cast()) };
    let mut acc = _mm256_setzero_ps();
    for i in 0..blocks {
        let b = &row[i * 108..][..108];
        let d = half(&ds[i * 2..]) * x.d[i];
        let (qs, qh, signs, scales) = (&b[0..64], &b[64..72], &b[72..104], &b[104..108]);
        let q8 = &x.q[i * QK..][..QK];
        let mut sums = _mm256_setzero_si256();
        for ib in (0..8).step_by(2) {
            let (q8a, q8b) = unsafe {
                (
                    _mm256_loadu_si256(q8[ib * 32..].as_ptr().cast()),
                    _mm256_loadu_si256(q8[ib * 32 + 32..].as_ptr().cast()),
                )
            };
            let g = |k: usize, h: usize, l: usize| {
                IQ3_S_GRID[usize::from(qs[k]) | ((usize::from(qh[h]) << (8 - l)) & 256)] as i32
            };
            let o = ib * 8;
            let a = _mm256_set_epi32(
                g(o + 7, ib, 7),
                g(o + 6, ib, 6),
                g(o + 5, ib, 5),
                g(o + 4, ib, 4),
                g(o + 3, ib, 3),
                g(o + 2, ib, 2),
                g(o + 1, ib, 1),
                g(o, ib, 0),
            );
            let o = o + 8;
            let c = _mm256_set_epi32(
                g(o + 7, ib + 1, 7),
                g(o + 6, ib + 1, 6),
                g(o + 5, ib + 1, 5),
                g(o + 4, ib + 1, 4),
                g(o + 3, ib + 1, 3),
                g(o + 2, ib + 1, 2),
                g(o + 1, ib + 1, 1),
                g(o, ib + 1, 0),
            );
            let s =
                |k: usize| u32::from_le_bytes([signs[k], signs[k + 1], signs[k + 2], signs[k + 3]]);
            let xa = unsafe { apply_signs(s(ib * 4), q8a, shuffle, bits) };
            let xc = unsafe { apply_signs(s(ib * 4 + 4), q8b, shuffle, bits) };
            let da = _mm256_maddubs_epi16(a, xa);
            let dc = _mm256_maddubs_epi16(c, xc);
            let ls1 = 2 * i16::from(scales[ib / 2] & 15) + 1;
            let ls2 = 2 * i16::from(scales[ib / 2] >> 4) + 1;
            sums = _mm256_add_epi32(
                sums,
                _mm256_add_epi32(
                    _mm256_madd_epi16(da, _mm256_set1_epi16(ls1)),
                    _mm256_madd_epi16(dc, _mm256_set1_epi16(ls2)),
                ),
            );
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sums), acc);
    }
    unsafe { hsum(acc) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq4_xs(row: &[u8], x: &Q8K) -> f32 {
    let blocks = row.len() / 136;
    assert_eq!(blocks * QK, x.q.len());
    let values = unsafe { _mm_loadu_si128(IQ4_NL_VALUES.as_ptr().cast()) };
    let m4 = _mm_set1_epi8(15);
    let mut acc = _mm256_setzero_ps();
    for i in 0..blocks {
        let b = &row[i * 136..][..136];
        let mut sh = u16::from_le_bytes([b[2], b[3]]);
        let (sl, qs) = (&b[4..8], &b[8..136]);
        let q8 = &x.q[i * QK..][..QK];
        let mut sums = _mm256_setzero_si256();
        for ib in (0..8).step_by(2) {
            let (bits1, bits2, q8a, q8b) = unsafe {
                (
                    _mm_loadu_si128(qs[ib * 16..].as_ptr().cast()),
                    _mm_loadu_si128(qs[ib * 16 + 16..].as_ptr().cast()),
                    _mm256_loadu_si256(q8[ib * 32..].as_ptr().cast()),
                    _mm256_loadu_si256(q8[ib * 32 + 32..].as_ptr().cast()),
                )
            };
            let expand = |v: __m128i| {
                _mm256_set_m128i(
                    _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(v, 4), m4)),
                    _mm_shuffle_epi8(values, _mm_and_si128(v, m4)),
                )
            };
            let (wa, wc) = (expand(bits1), expand(bits2));
            // Signed x signed through maddubs: move the weight's sign onto the input.
            let pa = _mm256_maddubs_epi16(_mm256_sign_epi8(wa, wa), _mm256_sign_epi8(q8a, wa));
            let pc = _mm256_maddubs_epi16(_mm256_sign_epi8(wc, wc), _mm256_sign_epi8(q8b, wc));
            let ls1 = ((i16::from(sl[ib / 2] & 15)) | ((sh << 4) & 0x30) as i16) - 32;
            let ls2 = ((i16::from(sl[ib / 2] >> 4)) | ((sh << 2) & 0x30) as i16) - 32;
            sh >>= 4;
            sums = _mm256_add_epi32(
                sums,
                _mm256_add_epi32(
                    _mm256_madd_epi16(pa, _mm256_set1_epi16(ls1)),
                    _mm256_madd_epi16(pc, _mm256_set1_epi16(ls2)),
                ),
            );
        }
        acc = _mm256_fmadd_ps(
            _mm256_set1_ps(half(b) * x.d[i]),
            _mm256_cvtepi32_ps(sums),
            acc,
        );
    }
    unsafe { hsum(acc) }
}

/// A raw GGUF Q2_K or Q3_K row (the NextN block's experts, kept as stored) dotted with the
/// quantized input: GGML's `ggml_vec_dot_q2_K_q8_K` / `q3_K` sums, without SIMD (the inner
/// loops of 16 vectorize).
pub fn dot_k(kind: TensorType, row: &[u8], x: &Q8K) -> f32 {
    let bb = match kind {
        TensorType::Q2K => 84,
        TensorType::Q3K => 110,
        _ => panic!("dot_k: unsupported type"),
    };
    assert_eq!(row.len() / bb * QK, x.q.len());
    let mut total = 0f32;
    for (i, b) in row.chunks_exact(bb).enumerate() {
        let q8 = &x.q[i * QK..][..QK];
        if kind == TensorType::Q2K {
            let (scales, packed) = (&b[..16], &b[16..80]);
            let (d, dmin) = (half(&b[80..]), half(&b[82..]));
            let (mut isum, mut msum) = (0i32, 0i32);
            let mut is = 0;
            for h in 0..2 {
                let q = &packed[h * 32..][..32];
                for j in 0..4 {
                    let shift = 2 * j;
                    for part in 0..2 {
                        let sc = scales[is];
                        is += 1;
                        let at = h * 128 + j * 32 + part * 16;
                        let (mut dot, mut sum) = (0i32, 0i32);
                        for l in 0..16 {
                            let x8 = i32::from(q8[at + l]);
                            dot += i32::from((q[part * 16 + l] >> shift) & 3) * x8;
                            sum += x8;
                        }
                        isum += i32::from(sc & 15) * dot;
                        msum += i32::from(sc >> 4) * sum;
                    }
                }
            }
            total += x.d[i] * (d * isum as f32 - dmin * msum as f32);
        } else {
            let (hmask, packed, raw) = (&b[..32], &b[32..96], &b[96..108]);
            let d = half(&b[108..]);
            let word = |k: usize| {
                u32::from_le_bytes([raw[4 * k], raw[4 * k + 1], raw[4 * k + 2], raw[4 * k + 3]])
            };
            let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
            let (a0, a1, tmp) = (word(0), word(1), word(2));
            let aux = [
                (a0 & k2) | ((tmp & k1) << 4),
                (a1 & k2) | (((tmp >> 2) & k1) << 4),
                ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
                ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4),
            ];
            let mut scales = [0i8; 16];
            for (k, w) in aux.iter().enumerate() {
                for (j, byte) in w.to_le_bytes().iter().enumerate() {
                    scales[k * 4 + j] = *byte as i8;
                }
            }
            let mut isum = 0i32;
            let mut is = 0;
            let mut m = 1u8;
            for h in 0..2 {
                let q = &packed[h * 32..][..32];
                for j in 0..4 {
                    let shift = 2 * j;
                    for part in 0..2 {
                        let sc = i32::from(scales[is]) - 32;
                        is += 1;
                        let at = h * 128 + j * 32 + part * 16;
                        let mut dot = 0i32;
                        for l in 0..16 {
                            let low = i32::from((q[part * 16 + l] >> shift) & 3);
                            let high = if hmask[part * 16 + l] & m != 0 { 0 } else { 4 };
                            dot += (low - high) * i32::from(q8[at + l]);
                        }
                        isum += sc * dot;
                    }
                    m <<= 1;
                }
            }
            total += x.d[i] * d * isum as f32;
        }
    }
    total
}

/// `dot_k` for every row of `rows` (raw rows of `row_bytes`).
pub fn matvec_k(kind: TensorType, rows: &[u8], row_bytes: usize, x: &Q8K, out: &mut [f32]) {
    for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
        *o = dot_k(kind, row, x);
    }
}

/// Checks a tensor type before it is routed here.
pub fn check(kind: TensorType) -> Result<()> {
    if supported(kind) {
        Ok(())
    } else {
        Err("expert_type_unsupported")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q2_K and Q3_K rows against their decoded values times the quantized input.
    #[test]
    fn k_quant_dots_match_decoding() {
        for (kind, bb) in [(TensorType::Q2K, 84usize), (TensorType::Q3K, 110)] {
            let blocks = 8;
            let mut s = 0x9e3779b97f4a7c15u64;
            let mut row: Vec<u8> = (0..blocks * bb)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    s as u8
                })
                .collect();
            // Sane f16 scales (a random bit pattern may be inf or NaN).
            for b in 0..blocks {
                let at = b * bb + if kind == TensorType::Q2K { 80 } else { 108 };
                row[at..at + 2].copy_from_slice(&0x2c00u16.to_le_bytes());
                if kind == TensorType::Q2K {
                    row[at + 2..at + 4].copy_from_slice(&0x2400u16.to_le_bytes());
                }
            }
            let x: Vec<f32> = (0..blocks * QK)
                .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
                .collect();
            let q = Q8K::new(&x);
            let mut w = vec![0f32; blocks * QK];
            kind.decode(&row, &mut w).unwrap();
            let xq: Vec<f32> = (0..x.len())
                .map(|i| q.d[i / QK] * f32::from(q.q[i]))
                .collect();
            let want: f32 = w.iter().zip(&xq).map(|(a, b)| a * b).sum();
            let got = dot_k(kind, &row, &q);
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "{kind:?}: {got} vs {want}"
            );
        }
    }

    fn random_blocks(kind: TensorType, blocks: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // GGUF blocks (with the leading scale).
        let bb = match kind {
            TensorType::Iq2S => 82,
            TensorType::Iq3S => 110,
            _ => 136,
        };
        let mut out = vec![0u8; bb * blocks];
        for b in out.chunks_exact_mut(bb) {
            for v in b.iter_mut() {
                *v = next() as u8;
            }
            // A modest finite scale: binary16 values in [2^-7, 2^-6).
            let d = 0x2000u16 | (next() % 0x400) as u16;
            b[..2].copy_from_slice(&d.to_le_bytes());
        }
        out
    }

    #[test]
    fn simd_dots_match_the_integer_reference_and_the_decoded_weights() {
        for (n, kind) in [TensorType::Iq2S, TensorType::Iq3S, TensorType::Iq4Xs]
            .into_iter()
            .enumerate()
        {
            let blocks = 16;
            let row = random_blocks(kind, blocks, 0x9e37_79b9 + n as u64);
            let x: Vec<f32> = (0..blocks * QK)
                .map(|i| ((i * 7919) % 255) as f32 / 37.0 - 3.4)
                .collect();
            let q = Q8K::new(&x);
            // The device's layout: bodies, scales apart (IQ4_XS whole).
            let (body, ds) = if kind == TensorType::Iq4Xs {
                (row.clone(), Vec::new())
            } else {
                let bb = row.len() / blocks;
                let mut body = Vec::new();
                let mut ds = Vec::new();
                for b in row.chunks_exact(bb) {
                    ds.extend_from_slice(&b[..2]);
                    body.extend_from_slice(&b[2..]);
                }
                (body, ds)
            };
            let simd = dot(kind, &body, &ds, &q);
            let scalar = dot_scalar(kind, &body, &ds, &q);
            // The decoded weights times the dequantized input, in f64.
            let mut w = vec![0f32; blocks * QK];
            kind.decode(&row, &mut w).unwrap();
            let exact: f64 = w
                .iter()
                .enumerate()
                .map(|(i, w)| f64::from(*w) * f64::from(q.d[i / QK]) * f64::from(q.q[i]))
                .sum();
            let scale: f64 = w.iter().map(|v| f64::from(v.abs())).sum::<f64>() * 3.5;
            assert!(
                ((simd - scalar) as f64).abs() <= 1e-5 * scale,
                "{kind:?}: simd {simd} scalar {scalar}"
            );
            assert!(
                (f64::from(scalar) - exact).abs() <= 1e-5 * scale,
                "{kind:?}: scalar {scalar} exact {exact}"
            );
        }
    }
}
