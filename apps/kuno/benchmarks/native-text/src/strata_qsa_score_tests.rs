// Public synthetic CPU source/data-flow oracles. No device or model is created.
use super::*;
#[test]
fn fixed_window_grid_covers_every_causal_score_once_through_full_context() {
    assert_eq!(WINDOW_SCORE_GRID, [256, 1, 1]);
    for capacity in [4usize, 64, 4096, 32768] {
        let blocks = capacity / 4;
        for count in 1..=8.min(capacity) {
            for start in [0, 1, 3, capacity - count]
                .into_iter()
                .filter(|&s| s + count <= capacity)
            {
                let top = (start + count) / 4;
                let mut visited = vec![0u8; count * blocks];
                let mut key_reads = vec![0u8; blocks];
                for group in 0..WINDOW_SCORE_GRID[0] {
                    if group * 8 > top {
                        continue;
                    }
                    for warp in 0..8 {
                        for b in (group * 8 + warp..=top).step_by(WINDOW_SCORE_GRID[0] * 8) {
                            if b >= blocks {
                                continue;
                            }
                            key_reads[b] += 1;
                            for t in 0..count {
                                let n = start + t + 1;
                                if b > n / 4 || (b == n / 4 && n % 4 == 0) {
                                    continue;
                                }
                                visited[t * blocks + b] += 1;
                            }
                        }
                    }
                }
                for t in 0..count {
                    let n = start + t + 1;
                    let visible = n / 4 + usize::from(n % 4 != 0);
                    for b in 0..blocks {
                        assert_eq!(visited[t * blocks + b], u8::from(b < visible));
                    }
                }
                assert!(key_reads.iter().all(|&n| n <= 1));
                assert_eq!(
                    key_reads.iter().map(|&n| usize::from(n)).sum::<usize>(),
                    (top + 1).min(blocks)
                );
            }
        }
    }
}
#[test]
fn reused_real_and_spare_keys_match_independent_per_query_f64_scores() {
    let real = (0..10)
        .map(|b| std::array::from_fn::<_, 128, _>(|d| ((d * 11 + b * 23) % 73) as f32 / 83. - 0.4))
        .collect::<Vec<_>>();
    let dead = std::array::from_fn::<_, 128, _>(|d| (d * 7 % 29) as f32 / 41. + 0.2);
    for start in 0..24 {
        for count in 1..=8 {
            let q = (0..count * 512)
                .map(|i| {
                    if (i % 512) / 128 == 2 {
                        -0.125
                    } else {
                        (i * 3 % 101) as f32 / 119. - 0.41
                    }
                })
                .collect::<Vec<_>>();
            for (b, kp) in real.iter().enumerate().take((start + count) / 4 + 1) {
                // One real/dead read per block, retained across every query.
                let kp4 = std::array::from_fn::<_, 32, _>(|l| {
                    std::array::from_fn::<_, 4, _>(|d| kp[l * 4 + d])
                });
                let kd4 = std::array::from_fn::<_, 32, _>(|l| {
                    std::array::from_fn::<_, 4, _>(|d| dead[l * 4 + d])
                });
                for t in 0..count {
                    let visible = start + t + 1;
                    let complete = visible / 4;
                    if b > complete || (b == complete && visible % 4 == 0) {
                        continue;
                    }
                    let cached = if b == complete { &kd4 } else { &kp4 };
                    let original = if b == complete { &dead } else { kp };
                    let mut got = 0f32;
                    let mut expected = 0f64;
                    for h in 0..4 {
                        let mut lanes = [0f32; 32];
                        for l in 0..32 {
                            let off = t * 512 + h * 128 + l * 4;
                            lanes[l] = ((cached[l][0] * q[off] + cached[l][1] * q[off + 1])
                                + cached[l][2] * q[off + 2])
                                + cached[l][3] * q[off + 3];
                        }
                        for offset in [16, 8, 4, 2, 1] {
                            let before = lanes;
                            for l in 0..32 {
                                lanes[l] = before[l] + before[l ^ offset];
                            }
                        }
                        got += lanes[0].max(0.);
                        expected += (0..128)
                            .map(|d| f64::from(original[d]) * f64::from(q[t * 512 + h * 128 + d]))
                            .sum::<f64>()
                            .max(0.);
                    }
                    assert!((f64::from(got) - expected).abs() < 3e-6);
                    if b == complete {
                        assert!(got < 32.);
                        assert_eq!(
                            (got + 1e9f32).to_bits(),
                            ((expected as f32) + 1e9f32).to_bits()
                        );
                    }
                }
            }
        }
    }
}
