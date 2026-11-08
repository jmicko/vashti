//! Exact size-preserving permutation used by paired public oracles and WGSL.
//! Parent/group offsets are deliberately outside the logical permutation.
pub(crate) fn half_offset(
    row: usize,
    column: usize,
    columns: usize,
    rows_per_tile: usize,
) -> Option<usize> {
    if ![8, 16].contains(&rows_per_tile) || columns == 0 || columns % 16 != 0 || column >= columns {
        return None;
    }
    let tile_row = row / rows_per_tile;
    tile_row
        .checked_mul(rows_per_tile)?
        .checked_mul(columns)?
        .checked_add((column / 16).checked_mul(rows_per_tile * 16)?)?
        .checked_add((row % rows_per_tile) * 16 + column % 16)
}
pub(crate) fn logical_offset(
    at: usize,
    columns: usize,
    rows_per_tile: usize,
) -> Option<(usize, usize)> {
    if ![8, 16].contains(&rows_per_tile) || columns == 0 || columns % 16 != 0 {
        return None;
    }
    let block = rows_per_tile.checked_mul(columns)?;
    let within = at % block;
    let row = (at / block)
        .checked_mul(rows_per_tile)?
        .checked_add(within % (rows_per_tile * 16) / 16)?;
    let column = (within / (rows_per_tile * 16)) * 16 + within % 16;
    Some((row, column))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_large_tile_address_is_bijective_with_original_extent_and_vector_boundaries() {
        for block in [8, 16] {
            for columns in [16, 32, 640, 2560] {
                for rows in [32, 128, 288, 512] {
                    let mut seen = vec![false; rows * columns];
                    for row in 0..rows {
                        for column in 0..columns {
                            let at = half_offset(row, column, columns, block).unwrap();
                            assert!(at < seen.len() && !seen[at]);
                            seen[at] = true;
                            assert_eq!(logical_offset(at, columns, block), Some((row, column)));
                            if column % 8 == 0 {
                                for i in 0..8 {
                                    assert_eq!(
                                        half_offset(row, column + i, columns, block),
                                        Some(at + i)
                                    );
                                }
                            }
                        }
                    }
                    assert!(seen.into_iter().all(|x| x));
                }
            }
        }
        assert_eq!(half_offset(0, 0, 0, 16), None);
        assert_eq!(half_offset(0, 0, 17, 16), None);
        assert_eq!(half_offset(0, 16, 16, 16), None);
        assert_eq!(half_offset(0, 0, 16, 32), None);
        assert_eq!(half_offset(usize::MAX, 0, 2560, 8), None);
    }
}
