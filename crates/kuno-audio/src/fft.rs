//! A small mixed-radix FFT (radix 2 and 5, any other factor by a direct transform) for the
//! codec's 1280-point inverse STFT.

/// The inverse DFT without the 1/n factor: `out[t] = sum_k x[k] e^{+2 pi i k t / n}`.
pub fn inverse(x: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    let p = [2, 5, 3].into_iter().find(|p| n % p == 0).unwrap_or(n);
    if p == n {
        return (0..n)
            .map(|t| {
                x.iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(re, im), (k, &(a, b))| {
                        let angle = 2.0 * std::f64::consts::PI * ((k * t) % n) as f64 / n as f64;
                        let (s, c) = angle.sin_cos();
                        (re + a * c - b * s, im + a * s + b * c)
                    })
            })
            .collect();
    }
    let m = n / p;
    // Decimation in time: the p interleaved subsequences, transformed, then combined.
    let parts: Vec<Vec<(f64, f64)>> = (0..p)
        .map(|r| inverse(&(0..m).map(|k| x[r + p * k]).collect::<Vec<_>>()))
        .collect();
    let mut out = vec![(0.0, 0.0); n];
    for k in 0..m {
        for q in 0..p {
            let t = k + m * q;
            let mut sum = (0.0, 0.0);
            for (r, part) in parts.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * ((r * t) % n) as f64 / n as f64;
                let (s, c) = angle.sin_cos();
                let (a, b) = part[k];
                sum.0 += a * c - b * s;
                sum.1 += a * s + b * c;
            }
            out[t] = sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_a_direct_transform() {
        for n in [1usize, 2, 5, 10, 64, 1280] {
            let x: Vec<(f64, f64)> = (0..n)
                .map(|i| ((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
                .collect();
            let fast = super::inverse(&x);
            for t in (0..n).step_by((n / 7).max(1)) {
                let mut want = (0.0, 0.0);
                for (k, &(a, b)) in x.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                    want.0 += a * angle.cos() - b * angle.sin();
                    want.1 += a * angle.sin() + b * angle.cos();
                }
                assert!(
                    (fast[t].0 - want.0).abs() < 1e-8 && (fast[t].1 - want.1).abs() < 1e-8,
                    "n {n} t {t}"
                );
            }
        }
    }
}
