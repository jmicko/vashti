//! Rust port of Strata native_expert.cpp's original IQ/Q2 CPU path.
//! MIT GGML 3cf03257f219afbe7334045ff7c6a06ac68c627d generic integer dots,
//! Q8_K/reference and x86 Q8_0 quantization. No file I/O or decoded model cache.
use crate::{Error, Result, TensorType, TensorView, quant};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};
pub const HOST_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_ROWS: usize = 8;
pub const MAX_ROUTES: usize = 80;
fn reject() -> Error {
    Error::InvalidFormat("native CPU expert window")
}
fn cancel(c: &AtomicBool) -> Result<()> {
    if c.load(Ordering::Acquire) {
        Err(reject())
    } else {
        Ok(())
    }
}

pub struct Geometry {
    columns: usize,
    ff: usize,
    kinds: [TensorType; 3],
    rows: [usize; 3],
    row_bytes: [usize; 3],
}
impl Geometry {
    pub fn new(t: [TensorView<'_>; 3]) -> Result<Self> {
        let d = t.map(|v| v.info().dimensions());
        if d.iter().any(|s| s.len() != 3)
            || d[0] != d[1]
            || d[2] != [d[0][1], d[0][0], d[0][2]]
            || d[0][0] != 2560
            || d[0][1] == 0
            || d[0][1] > 1024
            || d[0][2] != 512
        {
            return Err(reject());
        }
        let columns = d[0][0] as usize;
        let ff = d[0][1] as usize;
        let kinds = t.map(|v| v.info().dtype());
        if !matches!(
            kinds[0],
            TensorType::Iq2S | TensorType::Iq3S | TensorType::Iq3Xxs | TensorType::Iq4Xs
        ) || kinds[0] != kinds[1]
            || !matches!(kinds[2], TensorType::Iq4Nl | TensorType::Q2_0)
        {
            return Err(reject());
        }
        let rows = [ff, ff, columns];
        let widths = [columns, columns, ff];
        let mut row_bytes = [0; 3];
        for i in 0..3 {
            row_bytes[i] = kinds[i].row_bytes(widths[i] as u64)? as usize;
            if t[i].info().byte_len() != row_bytes[i] as u64 * rows[i] as u64 * 512 {
                return Err(reject());
            }
        }
        Ok(Self {
            columns,
            ff,
            kinds,
            rows,
            row_bytes,
        })
    }
    pub fn packed_bytes(&self, role: usize) -> Result<usize> {
        self.row_bytes
            .get(role)
            .zip(self.rows.get(role))
            .map(|(a, b)| a * b)
            .ok_or_else(reject)
    }
}
/// Caller authenticates the actual physical HOST cells and holds their joined
/// readonly maps through the pool call. Routes are original row*10+rank indices.
pub struct Group<'a> {
    pub routes: &'a [usize],
    pub packed: [&'a [u8]; 3],
}
struct Input {
    q: Zeroizing<Vec<i8>>,
    d: Zeroizing<Vec<f32>>,
    group: usize,
}
fn half(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 255) as i32 - 127 + 15;
    let frac = bits & 0x7fffff;
    let out = if exp >= 31 {
        sign | 0x7c00
    } else if exp < -10 {
        sign
    } else {
        let (m, shift) = if exp <= 0 {
            (frac | 0x800000, (14 - exp) as u32)
        } else {
            (frac, 13)
        };
        let base = m >> shift;
        let tail = m & ((1 << shift) - 1);
        let midpoint = 1 << (shift - 1);
        let rounded = base + u32::from(tail > midpoint || tail == midpoint && base & 1 != 0);
        sign + if exp <= 0 {
            rounded
        } else {
            (exp as u32) * 1024 + rounded
        }
    };
    quant::f16_to_f32(out as u16)
}
impl Input {
    fn new(x: &[f32], kind: TensorType, c: &AtomicBool) -> Result<Self> {
        let group = if matches!(kind, TensorType::Iq4Nl | TensorType::Q2_0) {
            32
        } else {
            256
        };
        if x.is_empty()
            || x.len() > 2560
            || !x.len().is_multiple_of(group)
            || x.iter().any(|v| !v.is_finite())
        {
            return Err(reject());
        }
        let mut q = Zeroizing::new(vec![0i8; x.len()]);
        let mut d = Zeroizing::new(vec![0f32; x.len() / group]);
        for (i, b) in x.chunks_exact(group).enumerate() {
            cancel(c)?;
            let mut maximum = Zeroizing::new(0f32);
            let mut signed = Zeroizing::new(0f32);
            for &v in b {
                if v.abs() > *maximum {
                    *maximum = v.abs();
                    *signed = v;
                }
            }
            let scale = if group == 256 {
                if *maximum == 0. { 0. } else { -127. / *signed }
            } else {
                if *maximum == 0. {
                    0.
                } else {
                    1. / (*maximum / 127.)
                }
            };
            d[i] = if scale == 0. {
                0.
            } else if group == 256 {
                1. / scale
            } else if kind == TensorType::Q2_0 {
                *maximum / 127.
            } else {
                half(*maximum / 127.)
            };
            if !d[i].is_finite() {
                return Err(reject());
            }
            for (j, &v) in b.iter().enumerate() {
                q[i * group + j] = if kind == TensorType::Q2_0 {
                    (scale * v).round().clamp(-127., 127.) as i8
                } else {
                    (scale * v).round_ties_even().clamp(-128., 127.) as i8
                };
            }
        }
        Ok(Self { q, d, group })
    }
}
fn h(b: &[u8]) -> f32 {
    quant::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}
fn signed(value: u8, signs: u32, bit: usize) -> i8 {
    if signs & (1 << bit) != 0 {
        -(value as i8)
    } else {
        value as i8
    }
}
fn scalar_dot(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| i32::from(a) * i32::from(b))
        .sum()
}
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn dot16(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    // SAFETY: caller checked AVX2 and these complete sixteen-byte slices.
    unsafe {
        let a = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().cast()));
        let b = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().cast()));
        let words = _mm256_madd_epi16(a, b);
        let mut out = [0i32; 8];
        _mm256_storeu_si256(out.as_mut_ptr().cast(), words);
        let sum = out.iter().sum();
        out.zeroize();
        sum
    }
}
fn idot(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && a.len().is_multiple_of(16) {
        #[allow(unsafe_code)]
        // SAFETY: exact chunks, guarded runtime ISA, no overread.
        return a
            .chunks_exact(16)
            .zip(b.chunks_exact(16))
            .map(|(a, b)| unsafe { dot16(a, b) })
            .sum();
    }
    scalar_dot(a, b)
}
/// Decode signed public grid codes, not scaled floating-point weight rows.
fn codes(kind: TensorType, b: &[u8], group: usize, out: &mut [i8; 32]) -> Result<i32> {
    use quant::{IQ2_S_GRID, IQ3_S_GRID, IQ3_XXS_GRID, IQ4_NL_VALUES};
    match kind {
        TensorType::Iq2S => {
            for o in 0..4 {
                let index = b[2 + group * 4 + o] as usize
                    | (((b[66 + group] as usize) << (8 - 2 * o)) & 0x300);
                let grid = IQ2_S_GRID[index].to_le_bytes();
                for j in 0..8 {
                    out[o * 8 + j] = signed(grid[j], u32::from(b[34 + group * 4 + o]), j);
                }
            }
            Ok(0)
        }
        TensorType::Iq3S => {
            for o in 0..4 {
                let high = b[66 + group] as usize;
                let first = 2 + group * 8 + o * 2;
                let a = IQ3_S_GRID[b[first] as usize | ((high << (8 - 2 * o)) & 256)].to_le_bytes();
                let d =
                    IQ3_S_GRID[b[first + 1] as usize | ((high << (7 - 2 * o)) & 256)].to_le_bytes();
                for (j, v) in a.into_iter().chain(d).enumerate() {
                    out[o * 8 + j] = signed(v, u32::from(b[74 + group * 4 + o]), j);
                }
            }
            Ok(i32::from(
                1 + 2 * ((b[106 + group / 2] >> (4 * (group % 2))) & 15),
            ))
        }
        TensorType::Iq3Xxs => {
            let at = 66 + group * 4;
            let word = u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
            for o in 0..4 {
                let s = (word >> (7 * o)) & 127;
                let s = s | ((s.count_ones() & 1) << 7);
                for part in 0..2 {
                    let grid = IQ3_XXS_GRID[b[2 + group * 8 + o * 2 + part] as usize].to_le_bytes();
                    for (j, v) in grid.into_iter().enumerate() {
                        out[o * 8 + part * 4 + j] = signed(v, s, part * 4 + j);
                    }
                }
            }
            Ok((1 + 2 * (word >> 28)) as i32)
        }
        TensorType::Iq4Xs | TensorType::Iq4Nl => {
            let at = if kind == TensorType::Iq4Xs {
                8 + group * 16
            } else {
                2
            };
            for j in 0..16 {
                out[j] = IQ4_NL_VALUES[(b[at + j] & 15) as usize];
                out[j + 16] = IQ4_NL_VALUES[(b[at + j] >> 4) as usize];
            }
            Ok(if kind == TensorType::Iq4Xs {
                let high = u16::from_le_bytes([b[2], b[3]]);
                i32::from((b[4 + group / 2] >> (4 * (group % 2))) & 15)
                    | i32::from(((high >> (2 * group)) & 3) << 4)
            } else {
                0
            })
        }
        _ => Err(reject()),
    }
}
fn dot_many(
    kind: TensorType,
    row: &[u8],
    input: &[Input],
    indices: &[usize],
    out: &mut [f32; 8],
    c: &AtomicBool,
) -> Result<()> {
    out.fill(0.);
    let elements = kind.block_elements() as usize;
    let bytes = kind.block_bytes() as usize;
    let columns = row.len() / bytes * elements;
    if row.is_empty()
        || !row.len().is_multiple_of(bytes)
        || indices.len() > 8
        || indices
            .iter()
            .any(|&i| input.get(i).is_none_or(|x| x.q.len() != columns))
    {
        return Err(reject());
    }
    for (block, b) in row.chunks_exact(bytes).enumerate() {
        cancel(c)?;
        let scale = h(b);
        if !scale.is_finite() {
            return Err(reject());
        }
        let mut integers = Zeroizing::new([0i32; 8]);
        for g in 0..elements / 32 {
            let first = block * elements + g * 32;
            let mut code = [0i8; 32];
            if kind == TensorType::Q2_0 {
                for (j, x) in code.iter_mut().enumerate() {
                    let pos = g * 32 + j;
                    *x = ((b[2 + pos / 4] >> (2 * (pos % 4))) & 3) as i8 - 1;
                }
            } else {
                let sub = codes(kind, b, g, &mut code)?;
                for (member, &i) in indices.iter().enumerate() {
                    let x = &input[i];
                    let q = &x.q[first..first + 32];
                    match kind {
                        TensorType::Iq2S => {
                            integers[member] += idot(&code[..16], &q[..16])
                                * i32::from(1 + 2 * (b[74 + g] & 15))
                                + idot(&code[16..], &q[16..]) * i32::from(1 + 2 * (b[74 + g] >> 4));
                        }
                        TensorType::Iq3S | TensorType::Iq3Xxs => {
                            integers[member] += idot(&code, q) * sub;
                        }
                        TensorType::Iq4Xs => {
                            out[member] += (scale * x.d[first / x.group] * (sub - 32) as f32)
                                * idot(&code, q) as f32;
                        }
                        TensorType::Iq4Nl => {
                            out[member] += (scale * x.d[first / x.group]) * idot(&code, q) as f32;
                        }
                        _ => return Err(reject()),
                    }
                }
                continue;
            }
            for (member, &i) in indices.iter().enumerate() {
                let x = &input[i];
                out[member] +=
                    (scale * x.d[first / x.group]) * idot(&code, &x.q[first..first + 32]) as f32;
            }
        }
        if matches!(
            kind,
            TensorType::Iq2S | TensorType::Iq3S | TensorType::Iq3Xxs
        ) {
            for (member, &i) in indices.iter().enumerate() {
                let x = &input[i];
                out[member] += (scale * x.d[block]) * integers[member] as f32;
            }
        }
    }
    let factor = match kind {
        TensorType::Iq2S => 0.125,
        TensorType::Iq3Xxs => 0.25,
        _ => 1.,
    };
    for x in out.iter_mut().take(indices.len()) {
        *x *= factor;
        if !x.is_finite() {
            return Err(reject());
        }
    }
    Ok(())
}
pub struct Pool {
    pool: rayon::ThreadPool,
}
pub struct Partials {
    values: Zeroizing<Vec<f32>>,
    mask: Zeroizing<[u32; 80]>,
}
impl Partials {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn mask(&self) -> &[u32; 80] {
        &self.mask
    }
}
impl Pool {
    pub fn new() -> Result<Self> {
        let threads = std::thread::available_parallelism()
            .map_err(|_| reject())?
            .get()
            .min(16);
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|_| reject())?,
        })
    }
    pub fn run(
        &self,
        g: &Geometry,
        jobs: &[Group<'_>],
        x: &[f32],
        count: usize,
        c: &AtomicBool,
    ) -> Result<Partials> {
        if !(1..=8).contains(&count) || jobs.len() > 80 || x.len() != count * g.columns {
            return Err(reject());
        }
        let mut mask = Zeroizing::new([0u32; 80]);
        for j in jobs {
            if j.routes.is_empty()
                || j.routes.len() > count
                || j.packed
                    .iter()
                    .enumerate()
                    .any(|(i, b)| b.len() != g.row_bytes[i] * g.rows[i])
            {
                return Err(reject());
            }
            let mut tokens = 0;
            for &r in j.routes {
                if r >= count * 10 || mask[r] != 0 || tokens & (1 << (r / 10)) != 0 {
                    return Err(reject());
                }
                mask[r] = 1;
                tokens |= 1 << (r / 10);
            }
        }
        let input: Vec<_> = x
            .chunks_exact(g.columns)
            .map(|x| Input::new(x, g.kinds[0], c))
            .collect::<Result<_>>()?;
        let mut gu = Zeroizing::new(vec![0f32; jobs.len() * g.ff * 8]);
        let mut down = Zeroizing::new(vec![0f32; jobs.len() * g.columns * 8]);
        let mut result = Zeroizing::new(vec![0f32; 80 * g.columns]);
        let allocation = (gu.capacity() + down.capacity() + result.capacity()) * 4 + 80 * 8 * 4096;
        if allocation > HOST_BYTES {
            return Err(Error::LimitExceeded("native CPU expert host envelope"));
        }
        self.pool.install(|| {
            gu.par_chunks_mut(8)
                .enumerate()
                .try_for_each(|(i, out)| -> Result<()> {
                    let j = &jobs[i / g.ff];
                    let feature = i % g.ff;
                    let mut indices = Zeroizing::new([0usize; 8]);
                    for (i, &r) in j.routes.iter().enumerate() {
                        indices[i] = r / 10;
                    }
                    let mut gate = Zeroizing::new([0f32; 8]);
                    let mut up = Zeroizing::new([0f32; 8]);
                    dot_many(
                        g.kinds[0],
                        &j.packed[0][feature * g.row_bytes[0]..(feature + 1) * g.row_bytes[0]],
                        &input,
                        &indices[..j.routes.len()],
                        &mut gate,
                        c,
                    )?;
                    dot_many(
                        g.kinds[1],
                        &j.packed[1][feature * g.row_bytes[1]..(feature + 1) * g.row_bytes[1]],
                        &input,
                        &indices[..j.routes.len()],
                        &mut up,
                        c,
                    )?;
                    for t in 0..j.routes.len() {
                        out[t] = crate::ops::silu(gate[t]) * up[t];
                        if !out[t].is_finite() {
                            return Err(reject());
                        }
                    }
                    Ok(())
                })
        })?;
        let h: Vec<Vec<Input>> = jobs
            .iter()
            .enumerate()
            .map(|(job, j)| {
                j.routes
                    .iter()
                    .enumerate()
                    .map(|(t, _)| {
                        let values = Zeroizing::new(
                            (0..g.ff)
                                .map(|f| gu[(job * g.ff + f) * 8 + t])
                                .collect::<Vec<_>>(),
                        );
                        Input::new(&values, g.kinds[2], c)
                    })
                    .collect()
            })
            .collect::<Result<_>>()?;
        self.pool.install(|| {
            down.par_chunks_mut(8)
                .enumerate()
                .try_for_each(|(i, out)| -> Result<()> {
                    let job = i / g.columns;
                    let column = i % g.columns;
                    let j = &jobs[job];
                    let mut indices = Zeroizing::new([0usize; 8]);
                    for (i, x) in indices.iter_mut().enumerate().take(j.routes.len()) {
                        *x = i;
                    }
                    let mut values = Zeroizing::new([0f32; 8]);
                    dot_many(
                        g.kinds[2],
                        &j.packed[2][column * g.row_bytes[2]..(column + 1) * g.row_bytes[2]],
                        &h[job],
                        &indices[..j.routes.len()],
                        &mut values,
                        c,
                    )?;
                    out.copy_from_slice(&*values);
                    Ok(())
                })
        })?;
        for (job, j) in jobs.iter().enumerate() {
            for (t, &r) in j.routes.iter().enumerate() {
                for col in 0..g.columns {
                    result[r * g.columns + col] = down[(job * g.columns + col) * 8 + t];
                }
            }
        }
        cancel(c)?;
        Ok(Partials {
            values: result,
            mask,
        })
    }
    /// GPU parent remains on its owning thread. CPU and parent always join
    /// before borrowed maps or private request owners leave this scope.
    pub fn alongside<R>(
        &self,
        g: &Geometry,
        jobs: &[Group<'_>],
        x: &[f32],
        count: usize,
        c: &AtomicBool,
        parent: impl FnOnce() -> Result<R>,
    ) -> Result<(Partials, R)> {
        std::thread::scope(|scope| {
            let cpu = scope.spawn(|| self.run(g, jobs, x, count, c));
            let gpu = std::panic::catch_unwind(std::panic::AssertUnwindSafe(parent));
            let cpu = cpu.join().map_err(|_| reject())??;
            Ok((cpu, gpu.map_err(|_| reject())??))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_quantizers_keep_signed_max_ties_and_the_distinct_q2_rounding() {
        let c = AtomicBool::new(false);
        let mut x = vec![0f32; 256];
        x[0] = 127.;
        x[1] = -127.;
        x[2] = 0.5;
        x[3] = -0.5;
        let k = Input::new(&x, TensorType::Iq3S, &c).unwrap();
        assert_eq!(k.d[0], -1.);
        assert_eq!(&k.q[..4], &[-127, 127, 0, 0]);
        let q2 = Input::new(&x[..32], TensorType::Q2_0, &c).unwrap();
        assert_eq!(q2.d[0], 1.);
        assert_eq!(&q2.q[..4], &[127, -127, 1, -1]);
        let nl = Input::new(&x[..32], TensorType::Iq4Nl, &c).unwrap();
        assert_eq!(&nl.q[..4], &[127, -127, 0, 0]);
        x[0] = f32::NAN;
        assert!(Input::new(&x, TensorType::Iq3S, &c).is_err());
    }
    fn block(kind: TensorType, salt: usize) -> Vec<u8> {
        let mut b = (0..kind.block_bytes() as usize)
            .map(|i| (i * 73 + salt * 19 + 7) as u8)
            .collect::<Vec<_>>();
        b[..2].copy_from_slice(&0x3400u16.to_le_bytes());
        b
    }
    #[test]
    fn native_integer_dots_match_independent_canonical_weights_with_quantized_inputs() {
        let c = AtomicBool::new(false);
        for kind in [
            TensorType::Iq2S,
            TensorType::Iq3S,
            TensorType::Iq3Xxs,
            TensorType::Iq4Xs,
            TensorType::Iq4Nl,
            TensorType::Q2_0,
        ] {
            for salt in 0..7 {
                let b = block(kind, salt);
                let n = kind.block_elements() as usize;
                let input = (0..8)
                    .map(|m| {
                        let x = (0..n)
                            .map(|i| (((i * 47 + m * 13) % 255) as f32 - 127.) / 32.)
                            .collect::<Vec<_>>();
                        Input::new(&x, kind, &c).unwrap()
                    })
                    .collect::<Vec<_>>();
                let mut weights = vec![0.; n];
                kind.decode(&b, &mut weights).unwrap();
                let mut got = [0.; 8];
                dot_many(kind, &b, &input, &[7, 2, 0], &mut got, &c).unwrap();
                for (member, &index) in [7, 2, 0].iter().enumerate() {
                    let x = &input[index];
                    let expected = (0..n)
                        .map(|i| {
                            f64::from(weights[i]) * f64::from(x.d[i / x.group]) * f64::from(x.q[i])
                        })
                        .sum::<f64>();
                    assert!(
                        (f64::from(got[member]) - expected).abs()
                            < 0.0002 + expected.abs() * 0.00002,
                        "dtype={} salt={salt}",
                        kind.gguf_id()
                    );
                }
            }
        }
    }
    #[test]
    fn native_integer_simd_is_exact_and_scoped_pool_preserves_sparse_rank_rows() {
        for salt in 0..29 {
            let a = (0..32).map(|i| (i * 59 + salt) as i8).collect::<Vec<_>>();
            let b = (0..32)
                .map(|i| (i * 17 + salt * 3) as i8)
                .collect::<Vec<_>>();
            assert_eq!(idot(&a, &b), scalar_dot(&a, &b));
        }
        let c = AtomicBool::new(false);
        let g = Geometry {
            columns: 256,
            ff: 32,
            kinds: [TensorType::Iq3S, TensorType::Iq3S, TensorType::Iq4Nl],
            rows: [32, 32, 256],
            row_bytes: [110, 110, 18],
        };
        let gate = block(TensorType::Iq3S, 3).repeat(32);
        let up = block(TensorType::Iq3S, 5).repeat(32);
        let down = block(TensorType::Iq4Nl, 7).repeat(256);
        let routes = (0..8).map(|i| i * 10).collect::<Vec<_>>();
        let jobs = [Group {
            routes: &routes,
            packed: [&gate, &up, &down],
        }];
        let x = (0..8 * 256)
            .map(|i| ((i % 29) as f32 - 14.) / 128.)
            .collect::<Vec<_>>();
        let called = AtomicBool::new(false);
        let pool = Pool::new().unwrap();
        let (result, ()) = pool
            .alongside(&g, &jobs, &x, 8, &c, || {
                called.store(true, Ordering::Release);
                Ok(())
            })
            .unwrap();
        assert!(called.load(Ordering::Acquire));
        for row in 0..80 {
            assert_eq!(result.mask[row], u32::from(row % 10 == 0));
            let values = &result.values[row * 256..(row + 1) * 256];
            assert!(values.iter().all(|x| x.is_finite()));
            if row % 10 != 0 {
                assert!(values.iter().all(|&x| x == 0.));
            }
        }
        assert!(
            pool.alongside(&g, &jobs, &x, 8, &c, || Err::<(), _>(reject()))
                .is_err()
        );
    }
}
