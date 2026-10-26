//! Independent, CPU-only operand-layout oracle for the packed LARGE candidate.
//!
//! Run beside `strata_large_tile_layout.rs` with:
//! `rustc --edition=2024 --test -O strata_large_tile_layout_oracle.rs -o /tmp/large-layout-oracle`
//! This validates address/bit preservation only; it is not GPU, quantization,
//! numerical, native-ownership, or performance acceptance. The reference order
//! is constructed by enumerating actual row/column tiles, without the candidate
//! address formula. Every value below is a fixed public synthetic coordinate.

#[path = "strata_large_tile_layout.rs"]
mod candidate;

#[cfg(test)]
mod tests {
    use super::candidate::{half_offset, logical_offset};

    const GU_COLUMNS: usize = 2560;
    const H_COLUMNS: usize = 640;
    const HALF_PER_IMAGE_SLOT: usize = 4_915_200;
    const DOWN_HALF_FIRST: usize = 3_276_800;
    const GUARD: usize = 17;
    const POISON: u32 = 0xdead_beef;

    /// Independent physical-order reference: visit reduction tiles before the
    /// rows inside each row tile, then visit the contiguous reduction values.
    fn tile_order(rows: usize, columns: usize, block: usize) -> Vec<(usize, usize)> {
        assert_eq!(rows % block, 0);
        assert_eq!(columns % 16, 0);
        let mut result = Vec::with_capacity(rows * columns);
        for first_row in (0..rows).step_by(block) {
            for first_column in (0..columns).step_by(16) {
                for row in first_row..first_row + block {
                    for column in first_column..first_column + 16 {
                        result.push((row, column));
                    }
                }
            }
        }
        result
    }

    fn pad32(n: usize) -> usize {
        n.div_ceil(32) * 32
    }

    fn pair_bits(source_token: usize, pair_column: usize) -> u32 {
        // A bit-preservation fixture, deliberately including high bits and
        // patterns that need not be finite FP16. No float conversion is made.
        let low = (source_token * 19 + pair_column * 7 + 0x8001) as u16;
        let high = (source_token * 31 + pair_column * 11 + 0x3c00) as u16;
        u32::from(low) | (u32::from(high) << 16)
    }

    #[test]
    fn independent_tile_enumeration_proves_bijection_inverse_and_vector_units() {
        for block in [8, 16] {
            for columns in [16, 32, 640, 2560] {
                for rows in [32, 64, 288, 512] {
                    let order = tile_order(rows, columns, block);
                    for (physical, &(row, column)) in order.iter().enumerate() {
                        assert_eq!(half_offset(row, column, columns, block), Some(physical));
                        assert_eq!(logical_offset(physical, columns, block), Some((row, column)));
                    }
                    // Both producers' vector types and the consumer's vec4<f16>
                    // must retain their exact original bit order and alignment.
                    for half_per_vector in [2, 4, 8] {
                        for (physical, vector) in order.chunks_exact(half_per_vector).enumerate() {
                            let (row, column) = vector[0];
                            assert_eq!(column % half_per_vector, 0);
                            assert_eq!(half_offset(row, column, columns, block).unwrap() / half_per_vector, physical);
                            for (within, &coordinate) in vector.iter().enumerate() {
                                assert_eq!(coordinate, (row, column + within));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn real_weight_images_keep_vec4_u32_store_and_vec4_f16_read_units_distinct() {
        // GU interleaves gate/up rows before the layout change. Tile packing
        // must preserve those row identities, including every image slot.
        let mut all_seen = vec![false; 8 * HALF_PER_IMAGE_SLOT / 8];
        for (down, rows, columns, role_first) in [
            (false, 1280, GU_COLUMNS, 0),
            (true, 2560, H_COLUMNS, DOWN_HALF_FIRST),
        ] {
            let order = tile_order(rows, columns, 8);
            for (physical_vector, original_eight) in order.chunks_exact(8).enumerate() {
                let (row, column) = original_eight[0];
                // Exact selected direct8 producer schedule, expressed in its
                // vec4<u32> units, against independently enumerated tile order.
                let i = row * (columns / 8) + column / 8;
                let packed_row = i / (columns / 8);
                let packed_col = i % (columns / 8);
                let producer_dest = (packed_row / 8) * columns
                    + (packed_col / 2) * 16 + (packed_row % 8) * 2 + packed_col % 2;
                assert_eq!(producer_dest, physical_vector);
                for slot in 0..8 {
                    let parent_half = slot * HALF_PER_IMAGE_SLOT + role_first;
                    let candidate_half = half_offset(row, column, columns, 8).unwrap();
                    // Dequant writes vec4<u32>, eight halves per element.
                    let producer_vector = parent_half / 8 + candidate_half / 8;
                    let expected_vector = parent_half / 8 + physical_vector;
                    assert_eq!(producer_vector, expected_vector);
                    assert!(!all_seen[producer_vector]);
                    all_seen[producer_vector] = true;
                    // Matrix loads bind vec4<f16>, four halves per element.
                    let consumer_vector = (parent_half + candidate_half) / 4;
                    assert_eq!(consumer_vector, 2 * producer_vector);
                    for (within, &(r, c)) in original_eight.iter().enumerate() {
                        assert_eq!((r, c), (row, column + within));
                        // This tests the row identity used by GU's original
                        // alternating gate/up decoder without changing it.
                        let original_projection = if down { 2 } else { r % 2 };
                        let original_source_row = if down { r } else { r / 2 };
                        assert_eq!(original_projection, if down { 2 } else { row % 2 });
                        assert_eq!(original_source_row, if down { row } else { row / 2 });
                    }
                }
            }
        }
        assert!(all_seen.into_iter().all(|x| x));
    }

    #[test]
    fn mixed_global_layer_gather_roundtrips_every_pair_and_clears_all_pad32_gutters() {
        let count = 512;
        // Exactly count*10 authentic route ordinals, all groups <= count.
        // Include every small/regular boundary and several partial LARGE tiles.
        let members = [1, 8, 31, 32, 255, 256, 257, 287, 288, 511, 512, 512, 512, 512, 512, 512, 122];
        assert_eq!(members.iter().sum::<usize>(), count * 10);
        let words_per_row = GU_COLUMNS / 2;
        let padded_rows = members.iter().map(|&n| pad32(n)).sum::<usize>();
        let mut written = vec![POISON; GUARD + padded_rows * words_per_row + GUARD];
        let mut expected = vec![0; padded_rows * words_per_row];
        let mut seen = vec![0_u8; expected.len()];
        let mut first = 0;
        let mut route_first = 0;
        for &real in &members {
            let padded = pad32(real);
            let group_base = first * words_per_row;
            // This deliberately translates the producer's actual u32 schedule.
            // Compare it to the independently enumerated expected order below.
            for member in 0..padded {
                for id_x in 0..words_per_row {
                    let at = if real >= 256 {
                        group_base
                            + (member / 16) * (16 * words_per_row)
                            + (id_x / 8) * 128
                            + (member % 16) * 8
                            + id_x % 8
                    } else {
                        group_base + member * words_per_row + id_x
                    };
                    assert!(at >= group_base && at < group_base + padded * words_per_row);
                    seen[at] += 1;
                    written[GUARD + at] = if member < real {
                        pair_bits((route_first + member) / 10, id_x)
                    } else { 0 };
                }
            }
            if real >= 256 {
                let order = tile_order(padded, GU_COLUMNS, 16);
                for (word, pair) in order.chunks_exact(2).enumerate() {
                    let (member, column) = pair[0];
                    assert_eq!(pair[1], (member, column + 1));
                    expected[group_base + word] = if member < real {
                        pair_bits((route_first + member) / 10, column / 2)
                    } else { 0 };
                }
            } else {
                for member in 0..real {
                    for pair_column in 0..words_per_row {
                        expected[group_base + member * words_per_row + pair_column] =
                            pair_bits((route_first + member) / 10, pair_column);
                    }
                }
            }
            // The consumers address global rows. This equivalence relies on
            // the original Pad32 seal; it must never be relaxed to any prefix.
            for member in 0..padded {
                for pair_column in 0..words_per_row {
                    let at = if real >= 256 {
                        half_offset(first + member, 2 * pair_column, GU_COLUMNS, 16).unwrap() / 2
                    } else {
                        (first + member) * words_per_row + pair_column
                    };
                    assert_eq!(written[GUARD + at], if member < real {
                        pair_bits((route_first + member) / 10, pair_column)
                    } else { 0 });
                }
            }
            first += padded;
            route_first += real;
        }
        assert_eq!(&written[GUARD..written.len() - GUARD], expected);
        assert!(seen.into_iter().all(|n| n == 1));
        assert!(written[..GUARD].iter().all(|&x| x == POISON));
        assert!(written[written.len() - GUARD..].iter().all(|&x| x == POISON));
    }

    #[test]
    fn activation_h_uses_cohort_prefix_and_initializes_its_entire_padded_owner() {
        let groups = [256, 257, 288, 511, 769];
        let words_per_row = H_COLUMNS / 2;
        let used_rows = groups.iter().map(|&n| pad32(n)).sum::<usize>();
        let mut actual = vec![POISON; 4096 * words_per_row];
        let mut expected = vec![POISON; actual.len()];
        let mut first = 0;
        for (group, &members) in groups.iter().enumerate() {
            let padded = pad32(members);
            for member in 0..padded {
                for pair_column in 0..words_per_row {
                    let original_i = member * words_per_row + pair_column;
                    let global_row = first + original_i / words_per_row;
                    let at = half_offset(global_row, (original_i % words_per_row) * 2, H_COLUMNS, 16).unwrap() / 2;
                    actual[at] = if member < members {
                        pair_bits(group * 1024 + member, pair_column)
                    } else { 0 };
                }
            }
            for (word, pair) in tile_order(padded, H_COLUMNS, 16).chunks_exact(2).enumerate() {
                let (row, column) = pair[0];
                expected[first * words_per_row + word] = if row < members {
                    pair_bits(group * 1024 + row, column / 2)
                } else { 0 };
            }
            first += padded;
        }
        assert_eq!(actual, expected);
        assert!(actual[used_rows * words_per_row..].iter().all(|&x| x == POISON));
        // A group's H prefix is the cohort prefix, not its global-X prefix.
        assert_ne!(half_offset(32, 16, H_COLUMNS, 16), half_offset(1024, 16, H_COLUMNS, 16));
    }

    #[test]
    fn flattened_activation_dispatch_zeroes_each_permuted_pad_word_once() {
        let groups = [256, 257, 511, 769];
        let max_members = *groups.iter().max().unwrap();
        let used_rows = groups.iter().map(|&n| pad32(n)).sum::<usize>();
        let words_per_row = H_COLUMNS / 2;
        let dispatch_threads = (used_rows * words_per_row).div_ceil(256) * 256;
        let stride = (max_members * words_per_row).div_ceil(256) * 256;
        let mut zero_visits = vec![0_u8; 4096 * words_per_row];
        // This follows the actual flattened X dispatch: find one group's
        // Pad32 interval, remap actual_id.x to that group's local id.x, then
        // run the zero loop BEFORE the real-row early return.
        for actual_id in 0..dispatch_threads {
            let mut prefix = 0;
            for &members in &groups {
                let padded = pad32(members);
                let first = prefix * words_per_row;
                let end = first + padded * words_per_row;
                if (first..end).contains(&actual_id) {
                    let local_id = actual_id - first;
                    let zero_first = members * words_per_row;
                    let zero_end = padded * words_per_row;
                    for at in (zero_first + local_id..zero_end).step_by(stride) {
                        // Exact packed_hidden_pair u32 schedule; compare below
                        // to physical-order enumeration, without this formula.
                        let row = at / words_per_row;
                        let pair = at % words_per_row;
                        let packed = (row / 16) * 5120 + (pair / 8) * 128
                            + (row % 16) * 8 + pair % 8;
                        zero_visits[first + packed] += 1;
                    }
                    break;
                }
                prefix += padded;
            }
        }
        let mut expected = vec![0_u8; zero_visits.len()];
        let mut prefix = 0;
        for &members in &groups {
            let padded = pad32(members);
            for (word, pair) in tile_order(padded, H_COLUMNS, 16).chunks_exact(2).enumerate() {
                if pair[0].0 >= members {
                    expected[prefix * words_per_row + word] = 1;
                }
            }
            prefix += padded;
        }
        assert_eq!(zero_visits, expected);
    }

    #[test]
    fn stride16_cmma_fragments_reconstruct_original_a_rows_and_b_columns() {
        for (columns, output_rows) in [(GU_COLUMNS, 1280), (H_COLUMNS, 2560)] {
            let weights = tile_order(output_rows, columns, 8);
            let input = tile_order(512, columns, 16);
            for k in (0..columns).step_by(16) {
                // Each prefix subgroup retains the four chronological A row
                // fragments and both B member fragments from the old eight C's.
                for out_row in (0..output_rows).step_by(32) {
                    for row_delta in [0, 8, 16, 24] {
                        let first = half_offset(out_row + row_delta, k, columns, 8).unwrap();
                        assert_eq!(first % 4, 0);
                        for a_row in 0..8 {
                            for a_column in 0..16 {
                                assert_eq!(weights[first + a_row * 16 + a_column],
                                    (out_row + row_delta + a_row, k + a_column));
                            }
                        }
                    }
                }
                for token in (0..512).step_by(32) {
                    for member_delta in [0, 16] {
                        let first = half_offset(token + member_delta, k, columns, 16).unwrap();
                        assert_eq!(first % 4, 0);
                        // B is K-by-member ColMajor: each matrix column is one
                        // contiguous reduction vector, with leading dimension16.
                        for b_column in 0..16 {
                            for b_row in 0..16 {
                                assert_eq!(input[first + b_column * 16 + b_row],
                                    (token + member_delta + b_column, k + b_row));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn parent_prefix_equivalence_requires_original_alignment_and_checked_extent() {
        for block in [8, 16] {
            for columns in [H_COLUMNS, GU_COLUMNS] {
                for prefix in (0..=56_832).step_by(32) {
                    for (row, column) in [(0, 0), (7, 8), (15, 16), (31, columns - 8)] {
                        assert_eq!(half_offset(prefix + row, column, columns, block),
                            half_offset(row, column, columns, block).map(|x| prefix * columns + x));
                    }
                }
            }
        }
        // Deliberately not equivalent for a non-tile-aligned parent. Retaining
        // prefix%32 guards is part of correctness, not redundant validation.
        assert_ne!(half_offset(8, 16, H_COLUMNS, 16),
            half_offset(0, 16, H_COLUMNS, 16).map(|x| 8 * H_COLUMNS + x));
        assert_eq!(half_offset(usize::MAX, 0, GU_COLUMNS, 8), None);
        assert_eq!(half_offset(16, 0, usize::MAX - 15, 16), None);
        assert_eq!(half_offset(0, GU_COLUMNS, GU_COLUMNS, 16), None);
    }
}
