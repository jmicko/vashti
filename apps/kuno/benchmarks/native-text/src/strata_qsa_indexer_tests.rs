// Public synthetic fixtures. CPU source/data-flow oracles, never GPU parity.
use super::*;
type Key = [f32; 128];
fn mean(rows: &[Key]) -> Key {
    std::array::from_fn(|d| {
        let mut sum = rows[0][d];
        for row in &rows[1..] {
            sum += row[d];
        }
        0.25f32.mul_add(sum, 0.)
    })
}
fn append(prefix: &[Key], incoming: &[Key], pooled: &mut [Key], dead: &mut Key) {
    let start = prefix.len();
    let mut tail = [[0.; 128]; 3];
    for (j, row) in tail.iter_mut().enumerate().take(start % 4) {
        *row = prefix[start / 4 * 4 + j];
    }
    for (t, row) in incoming.iter().enumerate() {
        let pos = start + t;
        let slot = pos % 4;
        if slot < 3 {
            tail[slot] = *row;
        }
        if pos == 0 {
            *dead = mean(&[*row; 4]);
            pooled[0] = *dead;
        } else if slot == 3 {
            pooled[pos / 4] = mean(&[tail[0], tail[1], tail[2], *row]);
        }
    }
}
#[test]
fn every_window_prefix_and_continuation_preserves_causal_pooled_keys() {
    let raw = (0..80)
        .map(|t| {
            std::array::from_fn(|d| {
                half::f16::from_f32(((t * 31 + d * 13) % 137) as f32 / 37. - 1.1).to_f32()
            })
        })
        .collect::<Vec<Key>>();
    for start in 0..32 {
        for count in 1..=8 {
            let mut full = vec![[0.; 128]; 21];
            let mut dead = [0.; 128];
            append(&[], &raw[..start], &mut full, &mut dead);
            append(
                &raw[..start],
                &raw[start..start + count],
                &mut full,
                &mut dead,
            );
            let expected_dead = mean(&[raw[0]; 4]);
            assert_eq!(dead, expected_dead);
            // A later query can complete a block that is still the current
            // query's incomplete tail. Direct scoring must use dead for it.
            for t in 0..count {
                let visible = start + t + 1;
                for (b, row) in full.iter().enumerate().take(visible / 4) {
                    assert_eq!(*row, mean(&raw[b * 4..b * 4 + 4]));
                }
                if visible % 4 != 0 {
                    assert_eq!(dead, expected_dead);
                }
            }
            for accepted in 1..=count {
                let end = start + accepted;
                let mut kept = full.clone();
                // Original suffix clear: only newly completed rejected blocks.
                kept[end / 4..(start + count) / 4].fill([0.; 128]);
                for (b, row) in kept.iter().enumerate().take(end / 4) {
                    assert_eq!(*row, mean(&raw[b * 4..b * 4 + 4]));
                }
                for row in &kept[end / 4..(start + count) / 4] {
                    assert_eq!(*row, [0.; 128]);
                }
                // A different continuation must not resurrect rejected keys.
                let replacement = &raw[64..68];
                append(&raw[..end], replacement, &mut kept, &mut dead);
                let mut reference = raw[..end].to_vec();
                reference.extend_from_slice(replacement);
                for (b, row) in kept.iter().enumerate().take(reference.len() / 4) {
                    assert_eq!(*row, mean(&reference[b * 4..b * 4 + 4]));
                }
                assert_eq!(dead, expected_dead);
            }
        }
    }
}
#[test]
fn native_two_level_subgroup_norm_matches_independent_f64_math() {
    for position in [0usize, 4, 128, 32764] {
        let means = std::array::from_fn::<_, 128, _>(|d| ((d * 19 % 53) as f32 - 26.) / 73.);
        let gamma = std::array::from_fn::<_, 128, _>(|d| 1. + (d % 7) as f32 / 17.);
        let mut lanes = [0f32; 256];
        for d in 0..128 {
            lanes[d] = means[d] * means[d];
        }
        for offset in [16, 8, 4, 2, 1] {
            let before = lanes;
            for d in 0..256 {
                lanes[d] = before[d] + before[(d / 32) * 32 + ((d % 32) ^ offset)];
            }
        }
        let mut sum = [0f32; 32];
        for w in 0..8 {
            sum[w] = lanes[w * 32];
        }
        for offset in [16, 8, 4, 2, 1] {
            let before = sum;
            for d in 0..32 {
                sum[d] = before[d] + before[d ^ offset];
            }
        }
        let scale = 1. / (sum[0] / 128. + 1e-6f32).sqrt();
        let expected_scale =
            1. / (means.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / 128. + 1e-6f64).sqrt();
        let mut y = std::array::from_fn::<_, 128, _>(|d| (scale * means[d]) * gamma[d]);
        let expected = std::array::from_fn::<_, 128, _>(|d| {
            expected_scale * f64::from(means[d]) * f64::from(gamma[d])
        });
        let theta = 1e7f32.powf(-2. / 64.);
        for pair in 0..32 {
            let angle = position as f32 * theta.powf(pair as f32);
            let c = angle.cos();
            let s = angle.sin();
            let a = y[pair];
            let b = y[pair + 32];
            y[pair] = a.mul_add(c, -b * s);
            y[pair + 32] = a.mul_add(s, b * c);
            // Isolate native norm/rotation arithmetic from backend pow/cos:
            // the F64 oracle uses the same public F32 angle coefficients.
            let lo = expected[pair] * f64::from(c) - expected[pair + 32] * f64::from(s);
            let hi = expected[pair] * f64::from(s) + expected[pair + 32] * f64::from(c);
            assert!((f64::from(y[pair]) - lo).abs() < 2e-6);
            assert!((f64::from(y[pair + 32]) - hi).abs() < 2e-6);
        }
        for d in 64..128 {
            assert!((f64::from(y[d]) - expected[d]).abs() < 2e-6);
        }
    }
}
#[test]
fn native_indexer_tail_and_pooled_addresses_stay_within_original_owners() {
    assert_eq!(Kernel::Indexer.lanes(), 256);
    assert_eq!(Kernel::IndexerFlags.lanes(), 64);
    assert_eq!(Kernel::DirectScore.lanes(), 256);
    for count in 1..=8 {
        for start in [0usize, 1, 2, 3, 63, 32768 - count] {
            let mut old_reads = 0;
            for j in 0..start % 4 {
                let cell = start / 4 * 4 + j;
                assert!(cell < start && start - cell <= 3);
                for d in 0..128 {
                    assert!(cell * CACHE_WORDS + INDEX_OFFSET + d < start * CACHE_WORDS);
                }
                old_reads += 1;
            }
            assert!(old_reads <= 3);
            for t in 0..count {
                let pos = start + t;
                if pos == 0 || pos % 4 == 3 {
                    assert!((pos / 4) * 128 + 127 < 32768 / 4 * 128);
                }
                for d in 0..128 {
                    assert!(t * CACHE_WORDS + INDEX_OFFSET + d < count * CACHE_WORDS);
                }
            }
        }
    }
}
