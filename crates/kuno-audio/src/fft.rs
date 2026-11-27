//! A small mixed-radix FFT (radix 2, 5 and 3, any other factor by a direct transform) for
//! the codec's 1280-point inverse STFT, with the twiddles of one size precomputed.

/// Inverse transforms of size `n` without the 1/n factor:
/// `out[t] = sum_k x[k] e^{+2 pi i k t / n}`.
pub struct Inverse {
    n: usize,
    /// `e^{+2 pi i j / n}` for `j < n`.
    twiddle: Vec<(f64, f64)>,
}

impl Inverse {
    pub fn new(n: usize) -> Self {
        let twiddle = (0..n)
            .map(|j| {
                let (s, c) = (2.0 * std::f64::consts::PI * j as f64 / n as f64).sin_cos();
                (c, s)
            })
            .collect();
        Self { n, twiddle }
    }

    pub fn run(&self, x: &[(f64, f64)]) -> Vec<(f64, f64)> {
        assert_eq!(x.len(), self.n);
        self.part(x)
    }

    /// A sub-transform of size `x.len()`, which divides `n`: its angle steps are multiples
    /// of the table's.
    fn part(&self, x: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let n = x.len();
        if n <= 1 {
            return x.to_vec();
        }
        let scale = self.n / n;
        let w = |j: usize| self.twiddle[(j % n) * scale];
        let p = [2, 5, 3]
            .into_iter()
            .find(|p| n.is_multiple_of(*p))
            .unwrap_or(n);
        if p == n {
            return (0..n)
                .map(|t| {
                    x.iter()
                        .enumerate()
                        .fold((0.0, 0.0), |(re, im), (k, &(a, b))| {
                            let (c, s) = w(k * t);
                            (re + a * c - b * s, im + a * s + b * c)
                        })
                })
                .collect();
        }
        let m = n / p;
        // Decimation in time: the p interleaved subsequences, transformed, then combined.
        let parts: Vec<Vec<(f64, f64)>> = (0..p)
            .map(|r| self.part(&(0..m).map(|k| x[r + p * k]).collect::<Vec<_>>()))
            .collect();
        let mut out = vec![(0.0, 0.0); n];
        for k in 0..m {
            for q in 0..p {
                let t = k + m * q;
                let mut sum = (0.0, 0.0);
                for (r, part) in parts.iter().enumerate() {
                    let (c, s) = w(r * t);
                    let (a, b) = part[k];
                    sum.0 += a * c - b * s;
                    sum.1 += a * s + b * c;
                }
                out[t] = sum;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_a_direct_transform() {
        for n in [1usize, 2, 5, 10, 64, 1280] {
            let x: Vec<(f64, f64)> = (0..n)
                .map(|i| ((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
                .collect();
            let fast = super::Inverse::new(n).run(&x);
            for t in (0..n).step_by((n / 7).max(1)) {
                let mut want = (0.0, 0.0);
                for (k, &(a, b)) in x.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                    want.0 += a * angle.cos() - b * angle.sin();
                    want.1 += a * angle.sin() + b * angle.cos();
                }
                let close = (fast[t].0 - want.0).abs() < 1e-8 && (fast[t].1 - want.1).abs() < 1e-8;
                assert!(close, "n {n} t {t}");
            }
        }
    }
}
