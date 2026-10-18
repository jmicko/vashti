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
    assert_eq!(CACHE_WORDS, 264);
    assert_eq!(INDEX_TAIL_WORDS, 384);
    assert_eq!(INDEX_WINDOW_WORDS, 1024);
    for count in 1..=8 {
        for start in [0usize, 1, 2, 3, 63, 32768 - count] {
            for j in 0..3 {
                for d in 0..128 {
                    assert!(j * 128 + d < INDEX_TAIL_WORDS);
                }
            }
            for t in 0..count {
                let pos = start + t;
                if pos == 0 || pos % 4 == 3 {
                    assert!((pos / 4) * 128 + 127 < 32768 / 4 * 128);
                }
                for d in 0..128 {
                    assert!(t * 128 + d < INDEX_WINDOW_WORDS);
                }
            }
        }
    }
}

// Full chronological history is an independent oracle for the bounded tail;
// restoring an accepted prefix must also preserve inactive slots exactly.
fn serial_tail(rows: &[Key]) -> [Key; 3] {
    let mut tail = [[0.; 128]; 3];
    for (cell, row) in rows.iter().enumerate() {
        if cell % 4 < 3 {
            tail[cell % 4] = row.map(|x| half::f16::from_f32(x).to_f32());
        }
    }
    tail
}
fn bounded_tail(snapshot: [Key; 3], raw: &[Key], start: usize, accepted: usize) -> [Key; 3] {
    let mut result = snapshot;
    if accepted != 0 {
        let last = start + accepted - 1;
        for (slot, dest) in result.iter_mut().enumerate() {
            if last >= slot {
                let cell = last - (last - slot) % 4;
                if cell >= start {
                    *dest = raw[cell - start].map(|x| half::f16::from_f32(x).to_f32());
                }
            }
        }
    }
    result
}
fn raw_keys(count: usize) -> Vec<Key> {
    (0..count)
        .map(|t| {
            std::array::from_fn(|d| {
                // Include halfway cases where preserving RTE through FP16 matters.
                if d % 17 == 0 {
                    1. + ((t + d) % 5) as f32 / 2048.
                } else {
                    ((t * 31 + d * 13) % 137) as f32 / 37. - 1.1
                }
            })
        })
        .collect()
}
#[test]
fn bounded_window_restore_matches_full_history_for_every_prefix_and_replacement() {
    let raw = raw_keys(32776);
    let starts = (0..32).chain([2047, 2048, 2049, 2051, 32760]);
    for start in starts {
        let snapshot = serial_tail(&raw[..start]);
        for count in 1..=8 {
            for accepted in 0..=count {
                let restored = bounded_tail(snapshot, &raw[start..start + count], start, accepted);
                assert_eq!(restored, serial_tail(&raw[..start + accepted]));
                let mut history = raw[..start + accepted].to_vec();
                history.extend_from_slice(&raw[32768..32776]);
                assert_eq!(
                    bounded_tail(restored, &raw[32768..32776], start + accepted, 8),
                    serial_tail(&history)
                );
            }
        }
    }
    // These are the actual GPU formulas, not a second unconnected host path.
    let prefix = include_str!("strata_prefix_qsa.wgsl");
    assert!(prefix.contains("last-(last-slot)%4u"));
    assert!(prefix.contains("tail_state[i]=accepted_tail(i)"));
    assert!(prefix.contains("bitcast<u32>(tail_state[i])!=bitcast<u32>(accepted_tail(i))"));
    assert!(!include_str!("strata_qsa_indexer.wgsl").contains("CACHE_WORDS"));
}
#[test]
fn bounded_large_prompt_tail_and_boundary_means_match_materialized_history() {
    let raw = raw_keys(32768);
    for count in [1, 2, 3, 4, 5, 7, 8, 9, 127, 4095, 4096] {
        for start in (0..8).chain([32768 - count]) {
            let snapshot = serial_tail(&raw[..start]);
            let incoming = &raw[start..start + count];
            assert_eq!(
                bounded_tail(snapshot, incoming, start, count),
                serial_tail(&raw[..start + count])
            );
            for block in start / 4..(start + count) / 4 {
                let gathered: [Key; 4] = std::array::from_fn(|j| {
                    let cell = block * 4 + j;
                    if cell < start {
                        snapshot[j]
                    } else {
                        incoming[cell - start].map(|x| half::f16::from_f32(x).to_f32())
                    }
                });
                let direct: Vec<_> = raw[block * 4..block * 4 + 4]
                    .iter()
                    .map(|r| r.map(|x| half::f16::from_f32(x).to_f32()))
                    .collect();
                assert_eq!(mean(&gathered), mean(&direct));
            }
        }
    }
    let source = Kernel::PromptTail.source();
    assert!(source.contains("last-(last-slot)%4u"));
    assert!(source.contains("snapshot[i]=old"));
    assert_eq!(Kernel::PromptTail.lanes(), 128);
}
