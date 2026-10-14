//! Carved prompt owners, borrowed from the three real admitted stage roots.
//! MIT Strata fb58e0 prefill.cpp::carve keeps GR/residual lifetime outside the
//! overlaid attention/MoE region. Here the working images are bounded to the
//! lower64 MiB and the connected GR intermediates occupy distinct upper views.
use super::*;
// Persistent view tables, construction vectors and the largest live QSA
// argument bundle are charged together; all GPU bytes remain in old roots.
pub(crate) const HOST_BYTES: usize = 16 * 1024;
const HALF: u64 = 64 * 1024 * 1024;
pub(crate) struct Storage {
    working: [Handle; 3],
    gr: [Handle; 4],   // normalized, down, low, up
    gdn: [Handle; 7],  // QKV, Z, H, readout, AB raw/controls/closing
    qsa: [Handle; 14], // five projections, Q/K/IQ, candidate, out, partial/stats, shared mean/valid
}
impl Storage {
    pub(crate) fn new(roots: [Handle; 3]) -> Result<Self> {
        if roots.iter().any(|h| {
            h.size_in_used() != 2 * HALF
                || h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
                || h.stream != roots[0].stream
        }) {
            return Err(Error::InvalidBufferLength);
        }
        let capacity = groups::CAPACITY;
        let carve = |root: &Handle, words: &[usize]| -> Result<Vec<Handle>> {
            let mut cursor = HALF;
            let mut views = Vec::with_capacity(words.len());
            for &n in words {
                // Native Alloc::take leaves256 bytes and aligns each cell to256.
                let bytes = (n * 4) as u64;
                let end = cursor.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
                let next = end.checked_add(511).ok_or(Error::ArithmeticOverflow)? & !255;
                if next > 2 * HALF {
                    return Err(Error::InvalidBufferLength);
                }
                views.push(root.clone().offset_start(cursor).offset_end(2 * HALF - end));
                cursor = next;
            }
            Ok(views)
        };
        let gr: [Handle; 4] = carve(&roots[0], &[10240, 320, 320, 10240].map(|w| w * capacity))?
            .try_into()
            .map_err(|_| Error::InvalidBufferLength)?;
        let gdn: [Handle; 7] = carve(
            &roots[1],
            &[10240, 6144, 10240, 6144, 96, 96, 48].map(|w| w * capacity),
        )?
        .try_into()
        .map_err(|_| Error::InvalidBufferLength)?;
        let qsa: [Handle; 14] = carve(
            &roots[1],
            &[
                12288 * capacity,
                512 * capacity,
                512 * capacity,
                512 * capacity,
                128 * capacity,
                6144 * capacity,
                512 * capacity,
                512 * capacity,
                640 * capacity,
                6144 * capacity,
                crate::strata_qsa::PARTIAL_WORDS * 32,
                crate::strata_qsa::STATS_WORDS * 32,
                1024 * 128,
                1024,
            ],
        )?
        .try_into()
        .map_err(|_| Error::InvalidBufferLength)?;
        let working = [
            roots[0].clone().offset_end(HALF),
            roots[1].clone().offset_end(HALF),
            roots[2].clone(),
        ];
        for (i, a) in gr.iter().enumerate() {
            if overlaps(a, &working[0])? || gr[..i].iter().any(|b| overlaps(a, b).unwrap_or(true)) {
                return Err(Error::InvalidBufferLength);
            }
        }
        for (i, a) in gdn.iter().enumerate() {
            if overlaps(a, &working[1])? || gdn[..i].iter().any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        for (i, a) in qsa.iter().enumerate() {
            if overlaps(a, &working[1])? || qsa[..i].iter().any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        Ok(Self {
            working,
            gr,
            gdn,
            qsa,
        })
    }
    pub(crate) fn working(&self) -> [Handle; 3] {
        self.working.clone()
    }
    pub(crate) fn gr(&self) -> [Handle; 4] {
        self.gr.clone()
    }
    pub(crate) fn gdn(&self) -> [Handle; 4] {
        std::array::from_fn(|i| self.gdn[i].clone())
    }
    pub(crate) fn gdn_controls(&self) -> [Handle; 3] {
        std::array::from_fn(|i| self.gdn[i + 4].clone())
    }
    pub(crate) fn qsa(&self) -> [Handle; 14] {
        self.qsa.clone()
    }
    pub(crate) fn output(&self, h: &Handle) -> bool {
        self.gr
            .iter()
            .chain(self.gdn.iter())
            .chain(self.qsa.iter())
            .any(|v| same_view(h, v))
    }
}
fn same_view(a: &Handle, b: &Handle) -> bool {
    a.stream == b.stream
        && a.clone().binding().memory.allocation_id() == b.clone().binding().memory.allocation_id()
        && a.offset_start.unwrap_or(0) == b.offset_start.unwrap_or(0)
        && a.offset_end.unwrap_or(0) == b.offset_end.unwrap_or(0)
        && a.size_in_used() == b.size_in_used()
}
/// Full allocation identity alone is insufficient once native phase regions
/// are carved. Admit only genuinely disjoint ranges of the same retained root.
pub(crate) fn overlaps(a: &Handle, b: &Handle) -> Result<bool> {
    if a.stream != b.stream {
        return Err(Error::InvalidBufferLength);
    }
    if a.clone().binding().memory.allocation_id() != b.clone().binding().memory.allocation_id() {
        return Ok(false);
    }
    let af = a.offset_start.unwrap_or(0);
    let bf = b.offset_start.unwrap_or(0);
    let ae = af
        .checked_add(a.size_in_used())
        .ok_or(Error::ArithmeticOverflow)?;
    let be = bf
        .checked_add(b.size_in_used())
        .ok_or(Error::ArithmeticOverflow)?;
    Ok(af < be && bf < ae)
}
