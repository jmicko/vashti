//! The C++ engine's two normal generators, value for value.
//!
//! `Mt19937` is PyTorch's CPU generator (the engine's `CPU_RNG`, which Kuno selects): an
//! MT19937 stream, 16 uniforms at a time turned into normals by Box-Muller in f32, with
//! the last group recomputed from fresh uniforms when the count is not a multiple of 16,
//! and a double-precision path below 16 values. `Philox` imitates PyTorch's CUDA generator
//! (Philox 4x32-10 with one Box-Muller value per counter), which the engine uses for the
//! noise added to conditioning latents.

pub struct Mt19937 {
    state: [u32; 624],
    next: usize,
    left: usize,
    gauss: Option<f64>,
}
impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed as u32;
        for j in 1..Self::N {
            let prev = state[j - 1];
            state[j] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(j as u32);
        }
        Self {
            state,
            next: 0,
            left: 1,
            gauss: None,
        }
    }
    fn twist(u: u32, v: u32) -> u32 {
        let mixed = (u & 0x8000_0000) | (v & 0x7fff_ffff);
        (mixed >> 1) ^ if v & 1 != 0 { 0x9908_b0df } else { 0 }
    }
    fn refill(&mut self) {
        let s = &mut self.state;
        let (n, m) = (Self::N, Self::M);
        for i in 0..n - m {
            s[i] = s[i + m] ^ Self::twist(s[i], s[i + 1]);
        }
        for i in n - m..n - 1 {
            s[i] = s[i + m - n] ^ Self::twist(s[i], s[i + 1]);
        }
        s[n - 1] = s[m - 1] ^ Self::twist(s[n - 1], s[0]);
        self.left = n;
        self.next = 0;
    }
    fn u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.refill();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
    fn uniform_f32(&mut self) -> f32 {
        // 24 bits of the draw over 2^24.
        (self.u32() & ((1 << 24) - 1)) as f32 * (1.0 / (1u32 << 24) as f32)
    }
    fn uniform_f64(&mut self) -> f64 {
        let high = u64::from(self.u32());
        let low = u64::from(self.u32());
        ((high << 32 | low) & ((1u64 << 53) - 1)) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn fill16(data: &mut [f32]) {
        for j in 0..8 {
            let u1 = 1.0f32 - data[j];
            let u2 = data[j + 8];
            let r = (-2.0f32 * u1.ln()).sqrt();
            let theta = 2.0f32 * std::f32::consts::PI * u2;
            data[j] = r * theta.cos();
            data[j + 8] = r * theta.sin();
        }
    }
    fn normal_f64(&mut self) -> f64 {
        if let Some(value) = self.gauss.take() {
            return value;
        }
        let u1 = self.uniform_f64();
        let u2 = self.uniform_f64();
        let r = (-2.0 * (-u2).ln_1p()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u1;
        self.gauss = Some(r * theta.sin());
        r * theta.cos()
    }
    /// `n` standard normals, as one `randn` call of the engine.
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        let mut data = vec![0f32; n];
        if n >= 16 {
            for v in &mut data {
                *v = self.uniform_f32();
            }
            for chunk in data.chunks_exact_mut(16) {
                Self::fill16(chunk);
            }
            if !n.is_multiple_of(16) {
                let tail = &mut data[n - 16..];
                for v in tail.iter_mut() {
                    *v = self.uniform_f32();
                }
                Self::fill16(tail);
            }
        } else {
            for v in &mut data {
                *v = self.normal_f64() as f32;
            }
        }
        data
    }
}

pub struct Philox {
    seed: u64,
    offset: u32,
}
impl Philox {
    pub fn new(seed: u64) -> Self {
        Self { seed, offset: 0 }
    }
    /// `n` standard normals, as one `randn` call of the engine; each call advances the
    /// offset by one.
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        const M: [u32; 2] = [0xD251_1F53, 0xCD9E_8D57];
        const W: [u32; 2] = [0x9E37_79B9, 0xBB67_AE85];
        const INV: f32 = 2.328_306_4e-10;
        const INV_2PI: f32 = 2.328_306_4e-10 * 6.283_185_5;
        let offset = self.offset;
        self.offset += 1;
        let (key0, key1) = (self.seed as u32, (self.seed >> 32) as u32);
        (0..n as u32)
            .map(|i| {
                let mut c = [offset, 0, i, 0];
                let (mut k0, mut k1) = (key0, key1);
                for round in 0..10 {
                    let v1 = u64::from(c[0]) * u64::from(M[0]);
                    let v2 = u64::from(c[2]) * u64::from(M[1]);
                    c = [
                        (v2 >> 32) as u32 ^ c[1] ^ k0,
                        v2 as u32,
                        (v1 >> 32) as u32 ^ c[3] ^ k1,
                        v1 as u32,
                    ];
                    if round < 9 {
                        k0 = k0.wrapping_add(W[0]);
                        k1 = k1.wrapping_add(W[1]);
                    }
                }
                let u = c[0] as f32 * INV + INV / 2.0;
                let v = c[1] as f32 * INV_2PI + INV_2PI / 2.0;
                // The engine's unqualified log, sqrt and sin take doubles; `s` is stored
                // as a float, the product rounds once at the end.
                let s = (-2.0 * f64::from(u).ln()).sqrt() as f32;
                (f64::from(s) * f64::from(v).sin()) as f32
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PyTorch: `torch.manual_seed(0)`, then `torch.randn(16)` (the vector path) or
    /// `torch.randn(4)` (the double-precision path below 16 values).
    #[test]
    fn mt19937_matches_torch() {
        let sixteen = [
            -1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920, -0.3160, -2.1152, 0.3223, -1.2633,
            0.3500, 0.3081, 0.1198, 1.2377, 1.1168, -0.2473,
        ];
        let four = [1.5410, -0.2934, -2.1788, 0.5684];
        for (want, got) in [
            (&sixteen[..], Mt19937::new(0).randn(16)),
            (&four[..], Mt19937::new(0).randn(4)),
        ] {
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-4, "{got:?}");
            }
        }
    }

    /// Values from the engine's own generators (`core/rng_mt19937.hpp`, `core/rng_philox.hpp`
    /// compiled alone): seed 42, 37 values (a 16-group tail and the small path), then 5.
    #[test]
    fn generators_match_the_engine() {
        let golden: Vec<Vec<f32>> =
            serde_json::from_str(include_str!("../../tests/fixtures/rng-golden.json")).unwrap();
        let mut mt = Mt19937::new(42);
        let mut philox = Philox::new(42);
        let got = [mt.randn(37), mt.randn(5), philox.randn(37), philox.randn(5)];
        for (i, (g, w)) in got.iter().zip(&golden).enumerate() {
            assert_eq!(g, w, "sequence {i}");
        }
    }
}
