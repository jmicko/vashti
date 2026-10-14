//! Carved prompt owners, borrowed from the three real admitted stage roots.
//! MIT Strata fb58e0 prefill.cpp::carve keeps GR/residual lifetime outside the
//! overlaid attention/MoE region. Here the working images are bounded to the
//! lower64 MiB and the connected GR intermediates occupy distinct upper views.
use super::*;
pub(crate) const HOST_BYTES: usize = 4096;
const HALF: u64 = 64 * 1024 * 1024;
pub(crate) struct Storage {
    working: [Handle; 3],
    gr: [Handle; 4],  // normalized, down, low, up
    gdn: [Handle; 4], // QKV, Z, H, raw recurrence output
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
        let carve = |root: &Handle, widths: [usize; 4]| -> Result<[Handle; 4]> {
            let mut cursor = HALF;
            let mut views = Vec::with_capacity(4);
            for width in widths {
                // Native Alloc::take leaves256 bytes and aligns each cell to256.
                let bytes = (capacity * width * 4) as u64;
                let end = cursor.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
                let next = end.checked_add(511).ok_or(Error::ArithmeticOverflow)? & !255;
                if next > 2 * HALF {
                    return Err(Error::InvalidBufferLength);
                }
                views.push(root.clone().offset_start(cursor).offset_end(2 * HALF - end));
                cursor = next;
            }
            views.try_into().map_err(|_| Error::InvalidBufferLength)
        };
        let gr = carve(&roots[0], [10240, 320, 320, 10240])?;
        let gdn = carve(&roots[1], [10240, 6144, 10240, 6144])?;
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
        Ok(Self { working, gr, gdn })
    }
    pub(crate) fn working(&self) -> [Handle; 3] {
        self.working.clone()
    }
    pub(crate) fn gr(&self) -> [Handle; 4] {
        self.gr.clone()
    }
    pub(crate) fn gdn(&self) -> [Handle; 4] {
        self.gdn.clone()
    }
    pub(crate) fn output(&self, h: &Handle) -> bool {
        self.gr
            .iter()
            .chain(self.gdn.iter())
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
