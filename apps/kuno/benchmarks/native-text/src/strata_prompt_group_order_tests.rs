use super::*;

pub(crate) fn packet(count: usize) -> Zeroizing<Vec<u32>> {
    let mut raw = Zeroizing::new(vec![0; LOGICAL + count * 10]);
    raw[..3].copy_from_slice(&[count as u32, 0, count as u32]);
    for token in 0..count {
        for rank in 0..10 {
            let local = if token < 48 { token } else if token % (rank + 3) == 0 { 0 } else { 1 + token % (rank + 2) };
            let logical = rank * 48 + local;
            let route = token * 10 + rank;
            raw[LOGICAL + route] = logical as u32;
            raw[MEMBERS + route] = ((logical * 13 + 17) % 512) as u32;
        }
    }
    raw
}
fn route_cells(map: &[u32], count: usize) -> Vec<u32> {
    let groups = validate(map, count).unwrap();
    let mut cells = vec![512; count * 10];
    for g in 0..groups {
        let first = map[OFFSETS + g] as usize;
        for &r in &map[MEMBERS + first..MEMBERS + first + map[LENGTHS + g] as usize] {
            assert_eq!(cells[r as usize], 512);
            cells[r as usize] = map[4 + g];
        }
    }
    assert!(cells.iter().all(|&c| c < 512));
    cells
}
#[test]
fn row_order_preserves_every_original_route_source_cell_and_within_expert_order() {
    for count in [1, 8, 32, 148, 257, ROUTE_CAPACITY] {
        let raw = packet(count);
        let old = group(&raw, count).unwrap();
        let new = group_for_prefill(&raw, count).unwrap();
        assert_eq!(route_cells(&old, count), route_cells(&new, count));
        assert_eq!(route_cells(&new, count), raw[MEMBERS..MEMBERS + count * 10]);
        let inv = ordered_inverse(&new, count).unwrap();
        for r in 0..count * 10 { assert_eq!(new[MEMBERS + inv[r] as usize], r as u32); }
        let mut keys = Vec::new();
        for g in 0..new[3] as usize {
            let first = new[OFFSETS + g] as usize;
            let route = new[MEMBERS + first] as usize;
            keys.push((new[LENGTHS + g], raw[LOGICAL + route]));
        }
        assert!(keys.windows(2).all(|p| p[0] < p[1]));
        assert_eq!(&old[..4], &new[..4]);
        #[cfg(not(feature = "strata-layer-gather"))]
        assert!(new[MEMBERS + count * 10..].iter().all(|&v| v == 0));
        #[cfg(feature = "strata-layer-gather")]
        {
            assert!(new[MEMBERS + count * 10..LAYER_PREFIX].iter().all(|&v| v == 0));
            validate_layer(&old, count).unwrap();
            validate_layer(&new, count).unwrap();
            assert!(new[LAYER_PREFIX + 513..].iter().all(|&v| v == 0));
        }
        if count >= 148 { assert_ne!(old.as_slice(), new.as_slice()); }
    }
}
#[test]
fn rank_ordered_weighted_results_are_bit_exact_for_both_whole_maps() {
    for count in [9, 148, 4096] {
        let raw = packet(count);
        let old = group(&raw, count).unwrap();
        let new = group_for_prefill(&raw, count).unwrap();
        let mut sums = Vec::new();
        for map in [&old, &new] {
            let inverse = ordered_inverse(map, count).unwrap();
            let mut partial = vec![[0f32; 8]; count * 10];
            for g in 0..map[3] as usize {
                let first = map[OFFSETS + g] as usize;
                for (i, &route) in map[MEMBERS + first..MEMBERS + first + map[LENGTHS + g] as usize].iter().enumerate() {
                    assert_eq!(map[4 + g], raw[MEMBERS + route as usize]);
                    for c in 0..8 {
                        let x = match (route as usize + c) % 6 {
                            0 => 16777216.0, 1 => -16777216.0, 2 => 1.0, 3 => -0.0,
                            _ => (map[4 + g] as f32 + (route / 10) as f32 + c as f32) / 32.0,
                        };
                        partial[first + i][c] = x;
                    }
                }
            }
            let mut out = Vec::new();
            for token in 0..count {
                for c in 0..8 {
                    let mut sum = 0f32;
                    for rank in 0..10 {
                        let r = token * 10 + rank;
                        sum = partial[inverse[r] as usize][c].mul_add((rank + 1) as f32 / 55.0, sum);
                    }
                    out.push(sum.to_bits());
                }
            }
            sums.push(out);
        }
        assert_eq!(sums[0], sums[1]);
    }
}
#[test]
fn production_policy_is_explicit_and_new_zeroizing_order_fits_original_scratch_budget() {
    let raw = packet(148);
    assert_eq!(group_for_prefill(&raw, 148).unwrap().as_slice(), group_with_order(&raw, 148, true).unwrap().as_slice());
    assert_eq!(group(&raw, 148).unwrap().as_slice(), group_with_order(&raw, 148, false).unwrap().as_slice());
    assert_eq!(ORDER_BYTES, 1024);
    // Includes BOTH raw/readback spans, both map/upload spans, inverse and
    // validation cells, original fixed arrays and the new full order stack.
    let conservative = 2 * RAW_WORDS * 4 + 2 * MAP_WORDS * 4 + ROUTES * 4 + 2 * ROUTES + 4 * 512 * size_of::<usize>() + ORDER_BYTES;
    assert!(conservative < HOST_BYTES);
    let src = include_str!("strata_global_moe.rs");
    assert_eq!(src.matches("groups::group_for_prefill(&raw, count)").count(), 1);
    let group = src.find("groups::group_for_prefill(&raw, count)").unwrap();
    let inverse = src[group..].find("groups::ordered_inverse(&map, count)").unwrap();
    let enqueue = src[group..].find("strata_prompt::experts::enqueue(").unwrap();
    assert!(inverse < enqueue);
    let consumer = include_str!("strata_prompt_ordered_shared_reduce.wgsl");
    assert!(consumer.contains("let route=row*10u+lane;let ordinal=inverse[route];"));
    assert!(consumer.contains("map[1540u+ordinal]==route"));
    assert!(consumer.contains("for(var rank=0u;rank<10u;rank++)"));
    assert!(consumer.contains("sum=fma(x,vec4<f32>(weight),sum);"));
}
#[test]
fn row_order_keeps_all_original_packet_rejections_and_deterministic_ties() {
    let raw = packet(148);
    let good = group_for_prefill(&raw, 148).unwrap();
    assert_eq!(good.as_slice(), group_for_prefill(&raw, 148).unwrap().as_slice());
    assert!(group_for_prefill(&raw, 0).is_err());
    assert!(group_for_prefill(&raw[..raw.len()-1], 148).is_err());
    for (at, value) in [(0, 147), (1, 1), (2, 149), (3, 1), (LOGICAL, 512), (MEMBERS, 512), (LOGICAL + 1, raw[LOGICAL]), (MEMBERS + 480, (raw[MEMBERS + 480] + 1) % 512)] {
        let mut bad = raw.clone(); bad[at] = value;
        assert!(group_for_prefill(&bad, 148).is_err(), "invalid field {at}");
        assert!(group(&bad, 148).is_err());
    }
}
