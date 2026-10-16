//! Original mapped embedding addresses occupy the unused tail of source table0.
//! No extra root or weight copy; the owning Sources/Weights retain every page.
use crate::strata_shared_weights::{self as shared, Tensor};
use kuno_text::{Error, Result};
pub(crate) const WORD_FIRST: usize = 3 * crate::strata_global_plan::EXPERTS * 4;
pub(crate) const PAGE_COUNT: usize = shared::ROWS.div_ceil(shared::TILE_ROWS);
pub(crate) fn encode(words: &mut [u32], pages: &[(u64, u64)]) -> Result<()> {
    if words.len() != crate::strata_global_sources::TABLE_BYTES as usize / 4
        || pages.len() != PAGE_COUNT
    {
        return Err(Error::InvalidBufferLength);
    }
    let tiles = shared::tiles(Tensor::Embedding);
    for ((&(address, owned), tile), entry) in pages
        .iter()
        .zip(&tiles)
        .zip(words[WORD_FIRST..WORD_FIRST + PAGE_COUNT * 4].chunks_exact_mut(4))
    {
        if address == 0
            || !address.is_multiple_of(16)
            || owned != tile.owned as u64
            || address.checked_add(owned).is_none()
        {
            return Err(Error::InvalidFormat("original mapped embedding address"));
        }
        entry.copy_from_slice(&[
            address as u32,
            (address >> 32) as u32,
            (tile.logical / 4) as u32,
            tile.rows as u32,
        ]);
    }
    Ok(())
}
const _: () =
    assert!(WORD_FIRST + PAGE_COUNT * 4 <= crate::strata_global_sources::TABLE_BYTES as usize / 4);
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_embedding_table_preserves_expert_words_and_covers_exact_original_rows() {
        let mut words = vec![0x55aa55aa; crate::strata_global_sources::TABLE_BYTES as usize / 4];
        let tiles = shared::tiles(Tensor::Embedding);
        let pages = tiles
            .iter()
            .enumerate()
            .map(|(i, t)| (0x1_0000_0000 + (i as u64) * 0x1000_0000, t.owned as u64))
            .collect::<Vec<_>>();
        encode(&mut words, &pages).unwrap();
        assert!(words[..WORD_FIRST].iter().all(|&v| v == 0x55aa55aa));
        assert!(
            words[WORD_FIRST + PAGE_COUNT * 4..]
                .iter()
                .all(|&v| v == 0x55aa55aa)
        );
        for token in 0..shared::ROWS {
            let page = token / shared::TILE_ROWS;
            let row = token % shared::TILE_ROWS;
            let at = WORD_FIRST + page * 4;
            let entry = &words[at..at + 4];
            assert_eq!(
                u64::from(entry[0]) | (u64::from(entry[1]) << 32),
                pages[page].0
            );
            assert!(row < entry[3] as usize);
            assert!((row + 1) * Tensor::Embedding.row_bytes() <= entry[2] as usize * 4);
            assert_eq!(entry[2] as usize * 4, tiles[page].logical);
        }
    }
    #[test]
    fn original_embedding_address_encoder_rejects_missing_extra_misaligned_or_wrong_owners() {
        let mut words = vec![0; crate::strata_global_sources::TABLE_BYTES as usize / 4];
        let pages = shared::tiles(Tensor::Embedding)
            .iter()
            .enumerate()
            .map(|(i, t)| (0x1_0000_0000 + (i as u64) * 0x1000_0000, t.owned as u64))
            .collect::<Vec<_>>();
        assert!(encode(&mut words, &pages[..PAGE_COUNT - 1]).is_err());
        let mut extra = pages.clone();
        extra.push(pages[0]);
        assert!(encode(&mut words, &extra).is_err());
        assert!(encode(&mut words[4..], &pages).is_err());
        for index in [0, PAGE_COUNT - 1] {
            for (address, owned) in [
                (0, pages[index].1),
                (pages[index].0 + 4, pages[index].1),
                (u64::MAX - 15, pages[index].1),
                (pages[index].0, pages[index].1 - 4),
            ] {
                let mut bad = pages.clone();
                bad[index] = (address, owned);
                assert!(encode(&mut words, &bad).is_err());
            }
        }
    }
}
