//! Carved prompt owners, borrowed from the three real admitted stage roots.
//! MIT Strata fb58e0 prefill.cpp::carve keeps GR/residual lifetime outside the
//! overlaid attention/MoE region. Here the working images are bounded to the
//! lower64 MiB and the connected GR intermediates occupy distinct upper views.
use super::*;
pub(crate) const HOST_BYTES: usize = 4096;
const HALF: u64 = 64 * 1024 * 1024;
pub(crate) struct Storage {
    working: [Handle; 3],
    gr: [Handle; 4], // normalized, down, low, up
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
        let mut cursor = HALF;
        let mut views = Vec::with_capacity(4);
        for width in [10240, 320, 320, 10240] {
            // Native Alloc::take leaves256 bytes and aligns each cell to256.
            let bytes = (capacity * width * 4) as u64;
            let end = cursor.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
            let next = end.checked_add(511).ok_or(Error::ArithmeticOverflow)? & !255;
            if next > 2 * HALF {
                return Err(Error::InvalidBufferLength);
            }
            views.push(
                roots[0]
                    .clone()
                    .offset_start(cursor)
                    .offset_end(2 * HALF - end),
            );
            cursor = next;
        }
        let gr: [Handle; 4] = views.try_into().map_err(|_| Error::InvalidBufferLength)?;
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
        Ok(Self { working, gr })
    }
    pub(crate) fn working(&self) -> [Handle; 3] {
        self.working.clone()
    }
    pub(crate) fn gr(&self) -> [Handle; 4] {
        self.gr.clone()
    }
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
