//! MIT Strata fb58e0 mtp.cpp::bind's original-order native draft vocabulary.
//! Separate immutable Q6_K rows are gathered once from the authenticated target
//! tensor. Public IDs preserve upstream tie order; private logits stay compact.
use super::*;

pub(crate) const ROWS: usize = 106_299;
pub(crate) const ROW_BYTES: usize = 2100;
pub(crate) const MAP_BYTES: usize = ROWS * 4;
pub(crate) const MAP_OWNED: usize = 524_288;
pub(crate) const INVERSE_OWNED: usize = 1_048_576;
pub(crate) const MAP_KIND: u32 = u32::MAX - 1;
pub(crate) const INVERSE_KIND: u32 = u32::MAX - 2;
pub(crate) const SHA256: &str = "b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91";
const RAW: &[u8; MAP_BYTES] =
    include_bytes!("../fixtures/strata-cjk-vocab/draft_vocab_cjk_original.bin");
fn reject() -> Error {
    Error::InvalidFormat("native CJK draft head owner")
}
pub(crate) fn selected(profile: MtpProfile) -> bool {
    crate::strata_intel_core::enabled() && profile == MtpProfile::StrataQ2
}
fn id(index: usize) -> Result<u32> {
    let at = index.checked_mul(4).ok_or(Error::ArithmeticOverflow)?;
    Ok(u32::from_le_bytes(
        RAW.get(at..at + 4)
            .ok_or_else(reject)?
            .try_into()
            .map_err(|_| reject())?,
    ))
}
fn inverse() -> Result<Vec<u32>> {
    let mut inverse = vec![u32::MAX; k::VOCAB];
    for at in 0..ROWS {
        let token = id(at)? as usize;
        let cell = inverse.get_mut(token).ok_or_else(reject)?;
        if *cell != u32::MAX {
            return Err(reject());
        }
        *cell = at as u32;
    }
    Ok(inverse)
}
fn plan() -> Vec<(usize, usize, usize, usize)> {
    (0..ROWS)
        .step_by(TILE_ROWS)
        .map(|first| {
            let rows = (ROWS - first).min(TILE_ROWS);
            let logical = rows * ROW_BYTES;
            (first, rows, logical, (logical + 4095) & !4095)
        })
        .collect()
}
pub(crate) fn public_bytes() -> u64 {
    plan().iter().map(|p| p.3 as u64).sum::<u64>() + (MAP_OWNED + INVERSE_OWNED) as u64
}
pub(crate) struct Head {
    pages: Vec<Page>,
    ids: Page,
    inverse: Page,
}
impl Head {
    pub(super) fn load(owner: &mut ResidentMtp<'_, '_>) -> Result<Self> {
        if !selected(owner.graph.profile()) || owner.public_english_vocab.is_some() {
            return Err(reject());
        }
        let inverse = inverse()?;
        let graph = owner.graph;
        let tensor = graph.tensor("output.weight")?;
        if tensor.info().dtype() != TensorType::Q6K
            || tensor.info().dimensions() != [2560, 248320]
            || tensor.info().row_bytes() as usize != ROW_BYTES
        {
            return Err(reject());
        }
        let mut pages = Vec::new();
        for (first, rows, logical, owned) in plan() {
            let mut source = Upload(vec![0; owned]);
            for row in 0..rows {
                // Copy original packed bytes, without dequantization/requantization.
                tensor.read_row(
                    id(first + row)? as u64,
                    &mut source.0[row * ROW_BYTES..(row + 1) * ROW_BYTES],
                )?;
            }
            validate_source(&source.0[..logical], quant_matvec::Q6_K)?;
            owner.verify_source()?;
            pages.push(owner.upload_page(
                source,
                logical,
                owned,
                first,
                rows,
                quant_matvec::Q6_K,
            )?);
        }
        let mut original = Upload(vec![0; MAP_OWNED]);
        original.0[..MAP_BYTES].copy_from_slice(RAW);
        let ids = owner.upload_page(original, MAP_BYTES, MAP_OWNED, 0, ROWS, MAP_KIND)?;
        let mut source = Upload(vec![0; INVERSE_OWNED]);
        for (word, value) in source.0[..k::VOCAB * 4].chunks_exact_mut(4).zip(&inverse) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        let inverse_page = owner.upload_page(
            source,
            k::VOCAB * 4,
            INVERSE_OWNED,
            0,
            k::VOCAB,
            INVERSE_KIND,
        )?;
        // Public-only startup readbacks authenticate map bytes, including padding.
        for (page, expected) in [
            (&ids, RAW.to_vec()),
            (
                &inverse_page,
                inverse.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        ] {
            let bytes = Bytes(
                owner
                    .client()?
                    .read_one(page.handle.clone())
                    .map_err(error)?,
            );
            if bytes.0.len() != page.owned_bytes
                || bytes.0[..expected.len()] != expected
                || bytes.0[expected.len()..].iter().any(|&v| v != 0)
            {
                return Err(reject());
            }
            owner.count_traffic(|t| {
                t.host_read_calls = t
                    .host_read_calls
                    .checked_add(1)
                    .ok_or(Error::ArithmeticOverflow)?;
                t.host_read_bytes = t
                    .host_read_bytes
                    .checked_add(page.owned_bytes as u64)
                    .ok_or(Error::ArithmeticOverflow)?;
                Ok(())
            })?;
        }
        owner.audit_io()?;
        let reserved = sample(owner.client()?)?;
        owner.ledger.completed_fence(reserved).map_err(error)?;
        let head = Self {
            pages,
            ids,
            inverse: inverse_page,
        };
        head.validate()?;
        Ok(head)
    }
    pub(super) fn validate(&self) -> Result<()> {
        let expected = plan();
        if self.pages.len() != expected.len() {
            return Err(reject());
        }
        for (p, (first, rows, logical, owned)) in self.pages.iter().zip(expected).chain([
            (&self.ids, (0, ROWS, MAP_BYTES, MAP_OWNED)),
            (&self.inverse, (0, k::VOCAB, k::VOCAB * 4, INVERSE_OWNED)),
        ]) {
            if (p.first, p.rows, p.logical_bytes, p.owned_bytes) != (first, rows, logical, owned)
                || p.handle.size() != owned as u64
                || p.handle.size_in_used() != owned as u64
                || p.handle.offset_start.unwrap_or(0) != 0
                || p.handle.offset_end.unwrap_or(0) != 0
            {
                return Err(reject());
            }
        }
        Ok(())
    }
    pub(super) fn handles(&self) -> impl Iterator<Item = Handle> + '_ {
        self.pages
            .iter()
            .chain([&self.ids, &self.inverse])
            .map(|p| p.handle.clone())
    }
    pub(super) fn maps(&self) -> [Handle; 2] {
        [self.ids.handle.clone(), self.inverse.handle.clone()]
    }
    pub(super) fn project(
        &self,
        client: &ComputeClient<WgpuRuntime>,
        input: Handle,
        planes: Handle,
        output: Handle,
        fault: Handle,
    ) -> Result<usize> {
        self.validate()?;
        crate::strata_dense::prepare(
            client,
            k::D,
            input,
            planes.clone(),
            fault.clone(),
            &matmul::BatchSpans::new(1, 0, 0, k::D, 0, ROWS),
        )?;
        for p in &self.pages {
            crate::strata_dense::project(
                client,
                quant_matvec::Q6_K,
                p.rows,
                k::D,
                p.handle.clone(),
                planes.clone(),
                output.clone(),
                fault.clone(),
                p.first,
                ROWS,
                1,
            )?;
        }
        Ok(self.pages.len() + 1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_cjk_ids_are_unique_and_inverse_preserves_exact_upstream_order() {
        let inv = inverse().unwrap();
        assert_eq!(inv.iter().filter(|&&v| v != u32::MAX).count(), ROWS);
        for index in 0..ROWS {
            assert_eq!(inv[id(index).unwrap() as usize], index as u32);
        }
        // This fixture is deliberately not sorted: sorting changes tie behavior.
        assert!((1..ROWS).any(|i| id(i - 1).unwrap() > id(i).unwrap()));
        let sorted = include_bytes!("../fixtures/strata-cjk-vocab/draft_vocab_cjk_sorted.bin");
        let mut actual: Vec<_> = (0..ROWS).map(|i| id(i).unwrap()).collect();
        actual.sort_unstable();
        let expected: Vec<_> = sorted
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected);
        // Like the existing bank-profile fixture, this Linux-only laboratory
        // authenticates the public upstream fixture with the installed digest.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/strata-cjk-vocab/draft_vocab_cjk_original.bin");
        let digest = std::process::Command::new("sha256sum")
            .arg(path)
            .output()
            .unwrap();
        assert!(digest.status.success());
        assert_eq!(&String::from_utf8(digest.stdout).unwrap()[..64], SHA256);
    }
    #[test]
    fn gathered_q6_page_plan_has_no_missing_rows_or_unpriced_padding() {
        let mut next = 0;
        let mut bytes = (MAP_OWNED + INVERSE_OWNED) as u64;
        for (first, rows, logical, owned) in plan() {
            assert_eq!(first, next);
            assert_eq!(logical, rows * ROW_BYTES);
            assert_eq!(owned % 4096, 0);
            assert!(owned >= logical && owned - logical < 4096);
            assert!(owned <= PUBLIC_SOURCE_BYTES);
            next += rows;
            bytes += owned as u64;
        }
        assert_eq!(next, ROWS);
        assert_eq!(bytes, public_bytes());
    }
}
